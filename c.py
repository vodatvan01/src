from models.modnet import MODNet
# from prettytable import PrettyTable
# modnet = MODNet()
# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
    
# count_parameters(modnet)

from dataloader import * 

def plot_images(boundaries, trimap, alpha, gt_detail):

    plt.figure(figsize=(16, 4))  # Điều chỉnh chiều rộng (16) và chiều cao (4) tùy ý

    # Vẽ hình ảnh đầu tiên (boundaries)
    plt.subplot(1, 4, 1)  # 1 hàng, 4 cột, vị trí 1
    plt.imshow(np.transpose(boundaries, (1, 2, 0)), cmap='gray')
    # plt.title('Boundaries')

    # Vẽ hình ảnh thứ hai (trimaps)
    plt.subplot(1, 4, 2)  # 1 hàng, 4 cột, vị trí 2
    plt.imshow(np.transpose(trimap, (1, 2, 0)), cmap='gray')
    # plt.title('Trimaps')

    # Vẽ hình ảnh thứ ba (alpha)
    plt.subplot(1, 4, 3)  # 1 hàng, 4 cột, vị trí 3
    plt.imshow(np.transpose(alpha, (1, 2, 0)), cmap='gray')
    plt.title('pred_detail')

    # Vẽ hình ảnh thứ tư (gt_detail)
    plt.subplot(1, 4, 4)  # 1 hàng, 4 cột, vị trí 4
    plt.imshow(np.transpose(gt_detail, (1, 2, 0)), cmap='gray')
    plt.title('pred_boundary_detail')

    plt.show()

def display_4_images(image1, image2, image3, image4):
    """
    Hiển thị 4 hình ảnh trong Matplotlib với cấu trúc 2 hàng và 2 cột.
    
    Args:
        image1, image2, image3, image4: Các mảng NumPy chứa dữ liệu hình ảnh.
    """
    # Khởi tạo subplot với kích thước 2 hàng và 2 cột
    fig, axes = plt.subplots(2, 2)

    # Hiển thị từng hình ảnh trên từng subplot
    axes[0, 0].imshow(image1)
    axes[0, 0].set_title(f'a : {image1.shape}')

    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 1].set_title(f'b : {image2.shape}')

    axes[1, 0].imshow(image3)
    axes[1, 0].set_title(f'c : {image3.shape}')

    axes[1, 1].imshow(image4, cmap='gray')
    axes[1, 1].set_title(f'd : {image4.shape}')

    # Ẩn các trục
    for ax in axes.flat:
        ax.axis('off')

    # Hiển thị subplot
    plt.tight_layout()
    plt.show()

def display_3_images(image1, image2, image3):
    fig, axes = plt.subplots(1, 2)


    axes[0].imshow(image2, cmap='gray')
    axes[0].set_title(f'alpha : {image2.shape}')

    axes[1].imshow(image3, cmap='gray')
    axes[1].set_title(f'trimap : {image3.shape}')

    # Ẩn các trục
    for ax in axes:
        ax.axis('off')

    # Hiển thị subplot
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def display_6_images(image1, image2, image3, image4, image5, image6):
    """
    Hiển thị 6 hình ảnh trong Matplotlib với cấu trúc 2 hàng và 3 cột.
    
    Args:
        image1, image2, image3, image4, image5, image6: Các mảng NumPy chứa dữ liệu hình ảnh.
    """
    # Khởi tạo subplot với kích thước 2 hàng và 3 cột
    fig, axes = plt.subplots(2, 3)

    # Hiển thị từng hình ảnh trên từng subplot
    axes[0, 0].imshow(image1)
    axes[0, 0].set_title(f'a : {image1.shape}')

    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 1].set_title(f'b : {image2.shape}')

    axes[0, 2].imshow(image3,  cmap='gray')
    axes[0, 2].set_title(f'c : {image3.shape}')

    axes[1, 0].imshow(image4)
    axes[1, 0].set_title(f'd : {image4.shape}')

    axes[1, 1].imshow(image5, cmap='gray')
    axes[1, 1].set_title(f'e : {image5.shape}')

    axes[1, 2].imshow(image6,  cmap='gray')
    axes[1, 2].set_title(f'f : {image6.shape}')

    # Ẩn các trục
    for ax in axes.flat:
        ax.axis('off')

    # Hiển thị subplot
    plt.tight_layout()
    plt.show()

# Sử dụng hàm
# display_6_images(image1, image2, image3, image4, image5, image6)

    
if __name__ == '__main__' :
     
    # load image path and mask path

    train_paths =  generate_paths_for_dataset('TRAIN') # => [image_path, mask_path]
    val500p_paths = generate_paths_for_dataset('VAL500P') # => [image_path, mask_path]
    val500np_paths = generate_paths_for_dataset('VAL500NP') # => [image_path, mask_path]

    # img = cv2.imread('P3M-10k/validation/P3M-500-NP/original_image/p_4aba7163.jpg')
    # alpha = cv2.imread('P3M-10k/validation/P3M-500-NP/mask/p_4aba7163.png',0)



    # load_image = LoadImage()
    # resize_by_short = ResizeByShort()
    # random_crop = RandomCrop()
    # flip_image = FlipImage()
    # gen_trimap = GenTrimap()
    # resize = Resize()

    # img, alpha = load_image(val500np_paths[102])
    # img, alpha = load_image(val500np_paths[0])
    # img, alpha = resize_by_short([img, alpha])
    # img, alpha = random_crop([img, alpha])
    # img, alpha = flip_image([img, alpha])
    # trimap = gen_trimap(alpha)
    # img_r, alpha_r, trimap_r = resize([img, alpha, trimap])
    
    # display_4_images(img, alpha, img, alpha)
    # display_3_images(img, alpha, trimap)
    # display_6_images(img, alpha, trimap, img_r, alpha_r, trimap_r )


    # image, alpha, trimap = transform('validation',test)

    train_dataset = MattingDataset(datasets = train_paths, phase= 'train', transform= MattingTransform())
    val500p_dataset = MattingDataset(datasets = val500p_paths, phase= 'validation', transform= MattingTransform())
    val500np_dataset = MattingDataset(datasets = val500np_paths, phase= 'validation', transform= MattingTransform())
        

    batch_size = 1
    train_dataloader = data.DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    val500p_dataloader = data.DataLoader(val500p_dataset, batch_size= batch_size, shuffle= True)
    val500np_dataloader = data.DataLoader(val500np_dataset, batch_size= batch_size, shuffle= True)
    
    dataload_dir = {
        "train" : train_dataloader,
        "val500p" : val500p_dataloader,
        "val500np" : val500np_dataloader,
    }

    batch_iterator = iter(dataload_dir['val500np'])

    images, alphas,trimaps = next(batch_iterator)
    boundaries = (trimaps < 0.5) + (trimaps > 0.5)
    gt_detail = torch.where(boundaries, trimaps, alphas)
 
    modnet = MODNet()
    pred_semantic, pred_detail, pred_matte = modnet(images, False)
    pred_boundary_detail = torch.where(boundaries, trimaps, pred_detail)
    gt_detail = torch.where(boundaries, trimaps, alphas)


    # detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
    test = pred_boundary_detail[0].detach() - gt_detail[0]
    # plot_images(boundaries[0], pred_detail[0].detach().numpy(), pred_boundary_detail[0].detach().numpy(), test)
    plot_images(boundaries[0], trimaps[0], pred_detail[0].detach().numpy(), pred_boundary_detail[0].detach().numpy())


    