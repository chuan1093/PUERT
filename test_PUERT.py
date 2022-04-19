import os
import glob
from time import time
import cv2
from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
from utils import *
from network.PUERT import PUERT

parser = ArgumentParser(description='')

parser.add_argument('--mark_str', type=str, default='Brain_LearnedM_PUERT', help='result directory')

parser.add_argument('--epoch_num', type=int, default=5000, help='epoch number of model')
parser.add_argument('--model_best', type=int, default=0, help='use net_params_best.pkl to test')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')

parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

parser.add_argument('--test_name', type=str, default='brain_test_50', help='name of test set')
parser.add_argument('--test_img_type', type=str, default='png', help='name of test set')
parser.add_argument('--saveimg', type=int, default=0, help='')

parser.add_argument('--layer_num', type=int, default=9, help='phase number')
parser.add_argument('--rb_num', type=int, default=2, help='')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')

parser.add_argument('--flag_1D', type=int, default=0, help='')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch_num = args.epoch_num
layer_num = args.layer_num
rb_num = args.rb_num
group_num = args.group_num
cs_ratio = args.cs_ratio
test_name = args.test_name
flag_1D = (args.flag_1D == 1)
desired_sparsity = cs_ratio/100.
sparse_ratio_ = 1.0 - cs_ratio / 100.

str_1D = '_1D' if flag_1D else ''
model_full_name = '%s_ratio_%d_layer_%d_group_%d%s' % (
    args.mark_str, cs_ratio, layer_num, group_num, str_1D)
mask_dir = os.path.join(args.model_dir, model_full_name)
mask_epoch_num = epoch_num

model_dir = os.path.join(args.model_dir, model_full_name)
result_dir = os.path.join(args.result_dir, model_full_name + '_teston_%s' % (test_name))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if epoch_num == -1:
    pre_model_dir = model_dir
    filelist = sorted(glob.glob('./%s/net_params_*.pkl' % pre_model_dir))
    assert len(filelist) > 0, 'epoch_num -1, but no model not exists!'
    int_list = []
    for i in range(len(filelist)):
        model_path = filelist[i]
        this_epoch = int(os.path.split(model_path)[-1].split('.')[0].split('_')[-1])
        int_list.append(this_epoch)
    epoch_num = max(int_list)
    print('epoch_num is -1, i.e., %d' % epoch_num)

log_file_name = "./%s/Log_TEST_%s_teston_%s.txt" % (
    args.log_dir, model_full_name, test_name)



# load data
test_dir = os.path.join(args.data_dir, test_name)
assert args.test_img_type == 'png' or args.test_img_type == 'tif', "not support such test_img_type"
filepaths = glob.glob(test_dir + '/*.' + args.test_img_type)
ImgNum = len(filepaths)

model = PUERT(layer_num, rb_num, desired_sparsity, sparse_ratio_, flag_1D)
model = nn.DataParallel(model)
model = model.to(device)

# Load pre-trained model with epoch number
model_path = '%s/net_params_%d.pkl' % (model_dir, epoch_num)
if args.model_best:
    model_path = '%s/net_params_best.pkl' % (model_dir)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path), strict=False)
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)

PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
TEST_TIME_All = np.zeros([1, ImgNum], dtype=np.float32)

print('\n')
print("MRI CS Reconstruction Start")
model.eval()
with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]
        Iorg = cv2.imread(imgName, 0)
        Icol = Iorg.reshape(1, 1, 256, 256) / 255.0

        gt = torch.from_numpy(Icol)
        gt = gt.type(torch.FloatTensor).to(device)

        test_start = time()
        model_output = model(gt)
        test_end = time()

        x_output = model_output[0]
        mask_matrix = model_output[1]

        Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)
        X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)

        rec_PSNR = psnr(X_rec * 255., Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec * 255., Iorg.astype(np.float64), data_range=255)
        rec_time = test_end - test_start

        print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f" % (
        img_no, ImgNum, imgName, (test_end - test_start), rec_PSNR, rec_SSIM))

        if args.saveimg == 1:
            im_rec_rgb = np.clip(X_rec * 255, 0, 255).astype(np.uint8)
            imgname_split = os.path.split(imgName)[-1]
            resultName = "%s_ratio_%d_epoch_%s_%s_PSNR_%.2f_SSIM_%.4f.png" \
                         % (imgname_split, cs_ratio, 'best' if args.model_best else str(epoch_num), model_full_name, rec_PSNR, rec_SSIM)
            savepath = os.path.join(result_dir, resultName)
            cv2.imwrite(savepath, im_rec_rgb)

        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM
        TEST_TIME_All[0, img_no] = rec_time

print('\n')
output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM for %s is %.2f/%.4f, Avg Rec Time is %.4f, Epoch is %d, Learned mask ratio is %.4f \n" % (
    cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(TEST_TIME_All), epoch_num, torch.mean(mask_matrix).item())
print(output_data)
output_file = open(log_file_name, 'a')
output_file.write(output_data)
output_file.close()

print("MRI CS Reconstruction End")
