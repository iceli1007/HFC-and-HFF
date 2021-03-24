"""
Implementation of Base SSGAN models.
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch_mimicry.nets.gan import gan
from torch.autograd import Variable
device = torch.device('cuda:5' if torch.cuda.is_available() else "cpu")
def distance(i, j, imageSize, r):
        dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

def mask_radial(batch_size,rows,cols, r):
    mask = np.zeros((batch_size,3, rows, cols))
    for t in range(batch_size):
        for i in range(rows):
            for j in range(cols):
                mask[t,0, i, j] = distance(i, j, imageSize=rows, r=r)
                mask[t,1, i, j] = distance(i, j, imageSize=rows, r=r)
                mask[t,2, i, j] = distance(i, j, imageSize=rows, r=r)
    return mask
def generateDataWithDifferentFrequencies_exchange(real,fake,r):
    # HFC
    real=real.mul(0.5).add(0.5)
    real_fft = torch.fft.fft2(real)
    real_fft_center = torch.fft.fftshift(real_fft)
    real_fd_low=real_fft_center*mask
    real_fd_high=real_fft_center*(1-mask)

    fake=fake.mul(0.5).add(0.5)
    fake_fft = torch.fft.fft2(fake)
    fake_fft_center = torch.fft.fftshift(fake_fft)
    fake_fd_low=fake_fft_center*mask
    fake_fd_high=fake_fft_center*(1-mask)

    real_main=real_fd_low+real_fd_high
    fake_main=real_fd_high+fake_fd_low


    real_main = torch.fft.ifftshift(real_main)
    real_main = torch.fft.ifft2(real_main)
    real_main=torch.abs(real_main).float()
    real_main=torch.clip(real_main,0,1)
    real_main=real_main.add(-0.5).mul(2)

    fake_main = torch.fft.ifftshift(fake_main)
    fake_main = torch.fft.ifft2(fake_main)
    fake_main=torch.abs(fake_main).float()
    fake_main=torch.clip(fake_main,0,1)
    fake_main=fake_main.add(-0.5).mul(2)
    return real_main,fake_main
def generateDataWithDifferentFrequencies_3Channel(image,r):
    # HFF
    image=image.mul(0.5).add(0.5)
    img_fft = torch.fft.fft2(image)
    img_fft_center = torch.fft.fftshift(img_fft)
    fd_low=img_fft_center*mask
    img_low_center = torch.fft.ifftshift(fd_low)
    img_low = torch.fft.ifft2(img_low_center)
    img_low=torch.abs(img_low).float()
    img_low=torch.clip(img_low,0,1)
    img_low=img_low.add(-0.5).mul(2)


    return img_low

R=20 #Radius of HFF and HFC
mask=mask_radial(64,32,32,R)
mask=torch.from_numpy(mask).cuda()

class SSGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for SSGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
        ss_loss_scale (float): Self-supervised loss scale for generator.
    """
    def __init__(self,
                 nz,
                 ngf,
                 bottom_width,
                 loss_type='hinge',
                 ss_loss_scale=0.2,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)
        self.ss_loss_scale = ss_loss_scale

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()
        real_images, real_labels = real_batch

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and logits
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)
        fake_images_low=generateDataWithDifferentFrequencies_3Channel(fake_images,R)
        #real_images_low,fake_images_low=generateDataWithDifferentFrequencies_exchange(real_images,fake_images,r=R)


        # Compute output logit of D thinking image real
        output, _ = netD(fake_images_low)

        # Compute GAN loss, upright images only.
        errG = self.compute_gan_loss(output)

        # Compute SS loss, rotates the images.
        errG_SS, _ = netD.compute_ss_loss(images=fake_images,
                                          scale=self.ss_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_SS
        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_SS', errG_SS, group='loss_SS')

        return log_data


class SSGANBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for SSGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.        
        ss_loss_scale (float): Self-supervised loss scale for discriminator.        
    """
    def __init__(self, ndf, loss_type='hinge', ss_loss_scale=1.0, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.num_classes = 4
        self.ss_loss_scale = ss_loss_scale

    def _rot_tensor(self, image, deg):
        r"""
        Rotation for pytorch tensors using rotation matrix. Takes in a tensor of (C, H, W shape).
        """
        if deg == 90:
            return image.transpose(1, 2).flip(1)

        elif deg == 180:
            return image.flip(1).flip(2)

        elif deg == 270:
            return image.transpose(1, 2).flip(2)

        elif deg == 0:
            return image

        else:
            raise ValueError(
                "Function only supports 90,180,270,0 degree rotation.")

    def _rotate_batch(self, images):
        r"""
        Rotate a quarter batch of images in each of 4 directions.
        """
        N, C, H, W = images.shape
        choices = [(i, i * 4 // N) for i in range(N)]

        # Collect rotated images and labels
        ret = []
        ret_labels = []
        degrees = [0, 90, 180, 270]
        for i in range(N):
            idx, rot_label = choices[i]

            # Rotate images
            image = self._rot_tensor(images[idx],
                                     deg=degrees[rot_label])  # (C, H, W) shape
            image = torch.unsqueeze(image, 0)  # (1, C, H, W) shape

            # Get labels accordingly
            label = torch.from_numpy(np.array(rot_label))  # Zero dimension
            label = torch.unsqueeze(label, 0)

            ret.append(image)
            ret_labels.append(label)

        # Concatenate images and labels to (N, C, H, W) and (N, ) shape respectively.
        ret = torch.cat(ret, dim=0)
        ret_labels = torch.cat(ret_labels, dim=0).to(ret.device)

        return ret, ret_labels

    def compute_ss_loss(self, images, scale):
        r"""
        Function to compute SS loss.

        Args:
            images (Tensor): A batch of non-rotated, upright images.
            scale (float): The parameter to scale SS loss by.

        Returns:
            Tensor: Scalar tensor representing the SS loss.
        """
        # Rotate images and produce labels here.
        images_rot, class_labels = self._rotate_batch(images=images)

        # Compute SS loss
        _, output_classes = self.forward(images_rot)

        err_SS = F.cross_entropy(input=output_classes, target=class_labels)

        # Scale SS loss
        err_SS = scale * err_SS

        return err_SS, class_labels

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Produce real images
        real_images, _ = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce fake images
        fake_images = netG.generate_images(num_images=batch_size,
                                           device=device).detach()
        real_images_low=generateDataWithDifferentFrequencies_3Channel(real_images,R)
        fake_images_low=generateDataWithDifferentFrequencies_3Channel(fake_images,R)
        #real_images_low,fake_images_low=generateDataWithDifferentFrequencies_exchange(real_images,fake_images,r=R)

        # Compute real and fake logits for gan loss
        output_real, _ = self.forward(real_images_low)
        output_fake, _ = self.forward(fake_images_low)

        # Compute GAN loss, upright images only.
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        # Compute SS loss, rotates the images.
        errD_SS, _ = self.compute_ss_loss(images=real_images,
                                          scale=self.ss_loss_scale)

        # Backprop and update gradients
        errD_total = errD + errD_SS
        errD_total.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('errD_SS', errD_SS, group='loss_SS')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data

    def advtrain_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one adversarial training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Produce real images
        real_images, _ = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce fake images
        fake_images = netG.generate_images(num_images=batch_size,
                                           device=device).detach()

        # Compute real and fake logits for gan loss
        output_real, _ = self.forward(real_images)
        output_fake, _ = self.forward(fake_images)

        
        #compute the adversarial samples of real and fake images.
        t=1
        real_value=torch.mean(output_real)
        fake_value=torch.mean(output_fake)
        fake_imgs_adv=fake_images.clone()
        real_imgs_adv=real_images.clone()
        real_imgs_adv=Variable(real_imgs_adv,requires_grad=True)
        fake_imgs_adv=Variable(fake_imgs_adv,requires_grad=True)
        #real_grad=Variable(real_grad,requires_grad=True)
        fake_output,_= self.forward(fake_imgs_adv)
        fake_output=fake_output.mean()
        fake_adv_loss = torch.abs(fake_output-real_value)
        #print(fake_adv_loss)
        #print(fake_adv_loss.requires_grad)
        #print(fake_imgs_adv.requires_grad)
        fake_grad=torch.autograd.grad(fake_adv_loss,fake_imgs_adv)
        fake_imgs_adv=fake_imgs_adv-fake_grad[0].clamp(-1*t,t)
        fake_imgs_adv=fake_imgs_adv.clamp(-1,1)
        real_output,_= self.forward(real_imgs_adv)
        real_output=real_output.mean()
        real_adv_loss = torch.abs(real_output-fake_value)
        real_grad=torch.autograd.grad(real_adv_loss,real_imgs_adv)
        real_imgs_adv=real_imgs_adv-real_grad[0].clamp(-1*t,t)
        real_imgs_adv=real_imgs_adv.clamp(-1,1)
        fake_adv_validity,_= self.forward(fake_imgs_adv.detach())
        real_adv_validity,_ = self.forward(real_imgs_adv)

        # Compute GAN loss, upright images only.
        errD = self.compute_gan_loss(output_real=real_adv_validity,
                                     output_fake=fake_adv_validity)

        # Compute SS loss, rotates the images.
        errD_SS, _ = self.compute_ss_loss(images=real_images,
                                          scale=self.ss_loss_scale)

        # Backprop and update gradients
        errD_total = errD + errD_SS
        errD_total.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=real_adv_validity,
                                       output_fake=fake_adv_validity)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('errD_SS', errD_SS, group='loss_SS')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data