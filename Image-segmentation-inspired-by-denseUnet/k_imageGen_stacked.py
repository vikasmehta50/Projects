import tifffile as tiff
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from functools import lru_cache
# import matplotlib 
# matplotlib.use('TkAgg')

class ImageHelper():
    
    def __init__(self, batch_size, tile_size, color_to_label):

        self.batch_size = batch_size
        self.tile_size = tile_size
        self.color_to_label = color_to_label
        self.mean = [86.60640368, 92.30221393, 85.75286437, 99.61715983]
        self.stddev = [34.79123667, 34.69273201, 36.12865668, 34.83905335]
        # self.norm_inputs = norm_inputs
        # self.isfitted = False

    def one_hot_labels(self, label_tile_rgb):
        
        if self.color_to_label == None:
            return label_tile_rgb
        
        labels_encoded = np.zeros(shape = [label_tile_rgb.shape[0], label_tile_rgb.shape[1], len(self.color_to_label)])

        for color, index in self.color_to_label.items():
            boolean_mask = ((np.sum(label_tile_rgb == color, axis = 2)) == label_tile_rgb.shape[-1])
            labels_encoded[boolean_mask, index] = 1

        return labels_encoded

    def fit_stats(self, image_path, sample_images = 5):

        print("Fitting Stats")
        image_ids = [os.path.basename(filename) for filename in glob(os.path.join(image_path, '*.tif'))]
        shuffle(image_ids)

        sample_images = len(image_ids) if (sample_images == None) else sample_images

        for i in range(sample_images):

            tmp_img = read_image(os.path.join(image_path, image_ids[i]))
            
            if i == 0:

                self.mean = np.empty(shape = (sample_images, tmp_img.shape[-1]))
                self.stddev = np.empty(shape = (sample_images, tmp_img.shape[-1]))
            
            # print('----------------*****----------------')
            # print('Filename: ', image_ids[i])
            
            self.mean[i] = np.mean(tmp_img, axis = (0,1))
            # print('mean: ', self.mean[i])

            self.stddev[i] = np.std(tmp_img, axis = (0,1))

        print("Stats Fitted")
        self.mean = np.mean(self.mean, axis = 0)
        self.mean = np.array([85,85,85,85])
        print("\nmean: ", self.mean)

        self.stddev = np.mean(self.stddev, axis = 0)
        self.stddev = np.array([35,35,35,35])
        print("std: ", self.stddev)

    def conv_pred_to_RGB(self, input_img):
        '''
        input_img = 4D Image (Convert to 3D)
        '''

        output_img = np.zeros(shape = (input_img.shape[1], input_img.shape[2], 3))
        input_img = np.argmax(input_img[0], axis = 2)

        for color, index in self.color_to_label.items():

            output_img[input_img == index] = color 

        return output_img

    def getImage(self, image_path, label_path,num_of_models):

        #print(image_path)
        #print(label_path)
        image_ids = sorted([os.path.basename(filename) for filename in glob(os.path.join(image_path, '*.nii.gz'))])
        label_ids = sorted([os.path.basename(filename) for filename in glob(os.path.join(label_path, '*.nii.gz'))])

        # print("all_ids: ",all_ids, " ", len(all_ids))
        # print("training_ids: ", image_ids, " ", len(image_ids), "\n\n")
        # print("label_ids: ", label_ids, " ", len(label_ids))
        
        # return
        #tiling logic
        
        is_stack_ready = False
        start_image_no = current_img = batch_counter = 0
        index = 0
        
        while True:

            # if is_stack_ready:
                
            #     if start_image_no + self.batch_size < len(t_img_stack):

            #         k = np.random.randint(1,5)

            #         yield np.rot90(t_img_stack[start_image_no: start_image_no + self.batch_size], k, axes = (1,2)), np.rot90(t_label_stack[start_image_no: start_image_no + self.batch_size], k, axes = (1,2))
                    
            #         start_image_no = start_image_no + self.batch_size
                
            #     else:

            #         # print("img_stack.shape : ", t_img_stack[start_image_no: len(image_ids)].shape)
            #         # print("label_stack.shape : ", t_label_stack[start_image_no: len(label_ids)].shape)
            #         # print("start_img_no: ", start_image_no)
            #         # print("len(image_ids): ", len(image_ids))
            #         # print("Yielding last batch of stack")

            #         k = np.random.randint(1,5)

            #         yield np.rot90(t_img_stack[start_image_no: len(t_img_stack)], k, axes = (1,2)), np.rot90(t_label_stack[start_image_no: len(t_label_stack)], k, axes = (1,2))
            #         # print("New Stack iteration begins\n\n")
            #         start_image_no = 0

            # else:

            img_filename = image_ids[index]
            lbl_filename = label_ids[index]
            
            start_row = start_col = 0
            print("Reading images.",os.path.join(image_path, img_filename),os.path.join(label_path, lbl_filename))
            tmp_image = read_image(os.path.join(image_path, img_filename))

            tmp_label = read_image(os.path.join(label_path, lbl_filename))

            # #Normalize inputs
            # if self.norm_inputs:

            #     if index == 0:
            #         self.fit_stats(image_path)

            #     tmp_image = tmp_image.astype(np.float32, copy = False)
            #     tmp_image = ( tmp_image - self.mean ) / self.stddev

                # print("mena: ", np.mean(tmp_image, axis = (0,1))) 

            #If image or label is 2D then convert it to 3D
            if len(tmp_label.shape) == 2:
                tmp_label = np.expand_dims(tmp_label, axis = 2)

            if len(tmp_image.shape) == 2:
                tmp_image = np.expand_dims(tmp_image, axis = 2)

            #if image is first - then create stack and perform some computations (n_images ....) - primary motive is to remove hardcoding
            if index == 0:

                in_channels = tmp_image.shape[-1]
                out_channels = len(self.color_to_label)
                image_height = tmp_image.shape[0]
                image_width = tmp_image.shape[1]

                eff_ts = self.tile_size // 2 #effective tile size / overlap factor
                # n_images_col = (image_width // eff_ts) if (image_width % eff_ts == 0) else ( (image_width // eff_ts) + 1 )
                # n_images_row = (image_height // eff_ts) if (image_height % eff_ts == 0) else ( (image_height // eff_ts) + 1 )
                n_images_col = len([i for i in range(0, tmp_image.shape[1], eff_ts) if (i + self.tile_size) < tmp_image.shape[1]]) + 1
                n_images_row = len([i for i in range(0, tmp_image.shape[0], eff_ts) if (i + self.tile_size) < tmp_image.shape[0]]) + 1

                # print("n_images_row: ", n_images_row)
                # print("n_images_col: ", n_images_col)

                n_images = ( n_images_col * n_images_row ) * len(image_ids)

                # print("n_tiles: ", n_images)

                t_img_stack = np.empty(shape = (self.batch_size, self.tile_size, self.tile_size, in_channels), dtype = np.float)
                t_label_stack = np.empty(shape = (self.batch_size, self.tile_size, self.tile_size, out_channels), dtype = np.float)

            while True:
                
                end_row = start_row + self.tile_size
                end_col = start_col + self.tile_size

                #Condition 1 and 4:
                if (end_col < tmp_image.shape[1]) and (end_row <= tmp_image.shape[0]):

                    #scan and stack
                    t_img_stack[batch_counter] = (tmp_image[start_row: end_row, start_col: end_col] - self.mean ) / self.stddev
                    t_label_stack[batch_counter] = self.one_hot_labels(tmp_label[start_row: end_row, start_col: end_col])

                    current_img += 1
                    batch_counter += 1

                    start_col = end_col - (self.tile_size // 2)

                #Condition 2 and 3:
                elif (end_col >= tmp_image.shape[1]) and (end_row < tmp_image.shape[0]):
                    
                    # tmp_start_col = start_col

                    start_col = tmp_image.shape[1] - self.tile_size
                    end_col = tmp_image.shape[1]

                    #scan and stack
                    t_img_stack[batch_counter] = (tmp_image[start_row: end_row, start_col: end_col] - self.mean ) / self.stddev
                    t_label_stack[batch_counter] = self.one_hot_labels(tmp_label[start_row: end_row, start_col: end_col])

                    current_img += 1
                    batch_counter += 1

                    # if(tmp_start_col + (self.batch_size // 2) < tmp_image.)
                    start_col = 0
                    start_row = end_row - (self.tile_size // 2)

                #Condition 5:
                elif end_row > tmp_image.shape[0]:

                    # print("Out of y")
                    start_row = tmp_image.shape[0] - self.tile_size
                    end_row = tmp_image.shape[0]

                    #scan and stack
                    t_img_stack[batch_counter] = (tmp_image[start_row: end_row, start_col: end_col] - self.mean ) / self.stddev
                    t_label_stack[batch_counter] = self.one_hot_labels(tmp_label[start_row: end_row, start_col: end_col])

                    current_img += 1
                    batch_counter += 1

                    start_col = end_col - (self.tile_size // 2)

                #Condition 6:
                elif (end_row == tmp_image.shape[0]) and (end_col >= tmp_image.shape[1]):

                    # print("Out of y")
                    start_col = tmp_image.shape[1] - self.tile_size
                    end_col = tmp_image.shape[1]

                    #scan and stack
                    t_img_stack[batch_counter] = (tmp_image[start_row: end_row, start_col: end_col] - self.mean ) / self.stddev
                    t_label_stack[batch_counter] = self.one_hot_labels(tmp_label[start_row: end_row, start_col: end_col])

                    current_img += 1
                    batch_counter += 1

                    k = np.random.randint(1,5)

                    if (batch_counter == self.batch_size) and (current_img != n_images):
                        #print("Data sent to training controller")
                        x_train = np.rot90(t_img_stack, k, axes = (1,2))
                        if True:
                            yield x_train, np.rot90(t_label_stack, k, axes = (1,2))
                        else:
                            print("entered")
                            yield (x_train for i in range(0,num_of_models)), np.rot90(t_label_stack, k, axes = (1,2))
                        #yield np.rot90(t_img_stack, k, axes = (1,2)),np.rot90(t_label_stack, k, axes = (1,2))
                        batch_counter = 0

                        # start_image_no = start_image_no + self.batch_size
                        
                    elif current_img == n_images:
                        x_train = np.rot90(t_img_stack[:batch_counter], k, axes = (1,2))
                        if True:
                            yield x_train, np.rot90(t_label_stack[:batch_counter], k, axes = (1,2))
                        else:
                            print("entered")
                            yield [x_train for i in range(0,num_of_models)], np.rot90(t_label_stack[:batch_counter], k, axes = (1,2))
                        #yield np.rot90(t_img_stack[:batch_counter], k, axes = (1,2)),np.rot90(t_label_stack[:batch_counter], k, axes = (1,2))
                        
                        # start_image_no 
                        batch_counter = current_img = 0
                        # is_stack_ready = True
                        print("Stack is Ready")
                        # break

                    # print("Filename: ", img_filename, "\nIndex: ", index, "\nNo of images Scanned: ", current_img)
                    index = (index + 1) % len(image_ids)
                    break

                else:
                    print("\n*******************\nNo condition is satisfied\n*******************\n")

                k = np.random.randint(1,5)

                #Check if batch is full or not
                if (batch_counter == self.batch_size) and (current_img != n_images):
                    x_train = np.rot90(t_img_stack, k, axes = (1,2))
                    if True:
                        yield x_train, np.rot90(t_label_stack, k, axes = (1,2))
                    else:
                        print("entered")
                        yield [x_train for i in range(0,num_of_models)], np.rot90(t_label_stack, k, axes = (1,2))
                    #yield np.rot90(t_img_stack, k, axes = (1,2)), np.rot90(t_label_stack, k, axes = (1,2))
                    batch_counter = 0
                    # start_image_no = start_image_no + self.batch_size
                    
                elif current_img == n_images:

                    print("\nBabuchak got executed")
                    x_train = np.rot90(t_img_stack[:batch_counter], k, axes = (1,2))
                    if True:
                        yield x_train, np.rot90(t_label_stack[:batch_counter], k, axes = (1,2))
                    else:
                        print("entered")
                        yield [x_train for i in range(0,num_of_models)], np.rot90(t_label_stack[:batch_counter], k, axes = (1,2))
                    #yield np.rot90(t_img_stack[:batch_counter], k, axes = (1,2)), np.rot90(t_label_stack[:batch_counter], k, axes = (1,2))

                    batch_counter = current_img = 0
                    # is_stack_ready = True
                    print("Stack is Ready")
                    break
                                    
        # print("all_ids: ",all_ids, " ", len(all_ids))
        # print("training_ids: ", training_ids, " ", len(training_ids))
        # print("Validation_ids: ", validation_ids, " ", len(validation_ids))

    # getImage()

@lru_cache(maxsize = 10000)
def read_image(image_path):

    tmp_image = tiff.imread(image_path)
    return tmp_image

if __name__=="__main__":

    color_to_label={(0,0,255):0, (0,255,0):1, (0,255,255):2, (255,0,0):3, (255,255,0):4, (255,255,255):5}
    ih = ImageHelper(5, 512, color_to_label)

    # ih.fit_stats('/tf/workspace/unet_seed/data/train_x', sample_images = 5)
    i = 1
    for x,y in ih.getImage('/tf/workspace/unet_seed/data/val_x','/tf/workspace/unet_seed/data/val_y'):
        print("x len: ", x.shape, "\ty len: ", y.shape, "\ti: ", i)
        i+=1
        # print("mean: ", np.mean(x, axis = (0,1,2)))
        # i+=1
        # if i == 4:
        #     break
        pass
    
    # a = np.random.randint()

    #     print("batch Generated", "  ", x.shape, "  ", y.shape)

    # t = getImage()
    # x,y = next(t)
    # # x,y = next(t)
    # import matplotlib.pyplot as plt
    # j = 16
    # plt.subplot(2,2,1)
    # plt.imshow(x[j, :, :, :3])

    # plt.subplot(2,2,2)
    # plt.imshow(x[j+1, :, :, :3])

    # plt.subplot(2,2,3)
    # plt.imshow(x[j+2, :, :, :3])

    # plt.subplot(2,2,4)
    # plt.imshow(x[j+3, :, :, :3])

    # plt.savefig(f'mygraph{j}.png')

# t = getImage()
# x,y = next(t)

# for i in range(32):

#     plt.imshow(x[i, :, :, :3])
#     plt.show()

# import matplotlib.pyplot as plt
# import tifffile as tiff
# tmp_image = tiff.imread("/root/Desktop/Dashansh_Project/data/4_Ortho_RGBIR/top_potsdam_4_11_RGBIR.tif")
# plt.imshow(tmp_image[:,:,:3])
# plt.show() in glob(os.path.join(label_path, '*.tif'))]