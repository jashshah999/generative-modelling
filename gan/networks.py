import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
	# TODO 1.1: Implement nearest neighbor upsampling + conv layer


	def __init__(
		self,
		input_channels,
		kernel_size=(3,3),
		n_filters=128,
		upscale_factor=2,
		padding=0,
	):
		super(UpSampleConv2D, self).__init__()
		# TODO 1.1: Setup the network layers
		self.input_channels = input_channels
		self.kernel_size = kernel_size
		self.n_filters = n_filters
		self.upscale_factor = upscale_factor
		self.padding = padding
		self.upscaled = nn.PixelShuffle(upscale_factor)
		# self.convLayer = nn.Conv2d(int(input_channels/upscale_factor),n_filters, kernel_size,stride=1, padding=padding)
		self.convLayer = nn.Conv2d(input_channels,n_filters,kernel_size,stride = 1, padding=padding)


	@torch.jit.script_method
	def forward(self, x):
		# TODO 1.1: Implement nearest neighbor upsampling
		# 1. Repeat x channel wise upscale_factor^2 times
		# 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
		# to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
		# 3. Apply convolution and return output


		# pixel_shuffle =  nn.PixelShuffle(self.upscale_factor*self.upscale_factor)
		# # print(pixel_shuffle)
		# x = pixel_shuffle(x)
		# x = self.convLayer
		# print(x)
		# upscale = self.upscale_factor(self.upscale_factor*self.upscale_factor)
		# upscale_factor_n = self.upscale_factor**2
		# upscale = self.upscaled(int(self.upscale_factor**2))

		# x = upscale(x)
		x = torch.cat([x,x,x,x],dim=1)
		# print(x.shape)
		# pixel_shuffle = nn.PixelShuffle(upscale_factor**2)
		# x = torch.cat([x,x,x,x], dim =1)
		x= self.upscaled(x)
		# x = x.unsqueeze
		# conv_layer = self.convLayer(self.input_channels,self.n_filters,self.kernel_size, stride=1, padding = self.padding)
		# print(x.shape)
		x = self.convLayer(x)

		# print(x.shape)

		return x

class DownSampleConv2D(torch.jit.ScriptModule):
	# TODO 1.1: Implement spatial mean pooling + conv layer

	def __init__(
		self, input_channels, kernel_size=(3,3), n_filters=128, downscale_ratio=2, padding=0
	):
		super(DownSampleConv2D, self).__init__()
		# TODO 1.1: Setup the network layers
		self.input_channels =input_channels
		self.kernel_size = kernel_size
		self.n_filters = n_filters
		self.downscale_ratio = downscale_ratio 
		self.padding = padding 


		# self.spatial_mean_pool = nn.AvgPool2d(kernel_size,stride =1,padding = padding)
		# self.conv_layer = nn.Conv2d(input_channels,n_filters,kernel_size,stride = 1,padding = padding)
		self.conv_layer = nn.Conv2d(in_channels=input_channels,out_channels=n_filters,kernel_size=kernel_size,stride=1,padding=padding,bias=True)
		self.pixel_unshuffle =nn.PixelUnshuffle(self.downscale_ratio)



	@torch.jit.script_method
	def forward(self, x):
		# TODO 1.1: Implement spatial mean pooling
		# 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
		# to form a (batch x channel * downscale_factor^2 x height x width) output
		# 2. Then split channel wise into (downscale_factor^2xbatch x channel x height x width) images
		# 3. Average across dimension 0, apply convolution and return output

		# x = self.pixel_unshuffle(x)
		# batch_temp = x.shape[0]
		# channel_temp = x.shape[1]/(self.downscale_ratio**2)
		# x = x.permute(0,2,3,1)
		# print("This is X shape")
		# print(x.shape)
		
		out = self.pixel_unshuffle(x)
		# x = F.pixel_unshuffle(x,downscale_factor=self.downscale_ratio)
		c = int(out.shape[1]//(self.downscale_ratio**2))
		out2 = torch.split(out, c, dim=1)
		# h, w = x[0].shape[2], x[0].shape[3]
		out_temp = torch.stack(out2, dim=0)
		# x = x.permute(0,2,1,3)
		# x = x.permute(0,3,1,2).contiguous()
		# x = x.permute((1/channel_temp)*batch_temp,1/(self.downscale_ratio**2),2,3)
		# x = x/(x.shape[0])
		# print(x.shape)
		# x = self.conv_layer(self.input_channels, self.n_filters, self.kernel_size, stride = 1, padding = self.padding)
		y= torch.mean(out_temp,dim=0)
		# print(self.conv_layer.weight.data.shape)
		final= self.conv_layer(y)
		return final
	
class ResBlockUp(torch.jit.ScriptModule):
	# TODO 1.1: Impement Residual Block Upsampler.
	"""
	ResBlockUp(
		(layers): Sequential(
			(0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(1): ReLU()
			(2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			(3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(4): ReLU()
			(5): UpSampleConv2D(
				(conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			)
		)
		(upsample_residual): UpSampleConv2D(
			(conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
		)
	"""

	def __init__(self, input_channels, kernel_size=(3,3), n_filters=128):
		super(ResBlockUp, self).__init__()
		# TODO 1.1: Setup the network layers
		self.input_channels = input_channels
		self.kernel_size = kernel_size
		self.n_filters = n_filters
		self.batch_norm_2d1 = nn.BatchNorm2d(input_channels,eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.batch_norm_2d2 = nn.BatchNorm2d(n_filters,eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True)
		# self.Relu = nn.ReLU()
		self.Conv2d = nn.Conv2d(input_channels,n_filters,kernel_size=(3,3), stride =1, padding =1, bias = False)
		self.upsample_conv2d = UpSampleConv2D(n_filters, n_filters=n_filters, kernel_size=(3,3), padding=1)
		self.resblockup = nn.Sequential(self.batch_norm_2d1, 
				  nn.ReLU(),self.Conv2d,
				  self.batch_norm_2d2,
				  nn.ReLU(),
				  self.upsample_conv2d)
	
		self.upsample_residual = UpSampleConv2D(input_channels, n_filters=n_filters, kernel_size=(1,1))
	@torch.jit.script_method
	def forward(self, x):
		# TODO 1.1: Forward through the layers and implement a residual connection.
		# Make sure to upsample the residual before adding it to the layer output.
		# x = self.resblockup(x) + self.upsample_residual(x)
		x1 = self.resblockup(x)
		x2 = self.upsample_residual(x)


		return x1+x2

class ResBlockDown(torch.jit.ScriptModule):
	# TODO 1.1: Impement Residual Block Downsampler.
	"""
	ResBlockDown(
		(layers): Sequential(
			(0): ReLU()
			(1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			(2): ReLU()
			(3): DownSampleConv2D(
				(conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			)
		)
		(downsample_residual): DownSampleConv2D(
			(conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
		)
	)
	"""
	def __init__(self, input_channels, kernel_size=(3,3), n_filters=128):
		super(ResBlockDown, self).__init__()
		# TODO 1.1: Setup the network layers
		self.input_channels = input_channels
		self.kernel_size = kernel_size
		self.n_filters = n_filters
		# self.Relu = nn.ReLU()
		self.Conv2d = nn.Conv2d(in_channels=input_channels,out_channels = n_filters,kernel_size=(3,3), stride =1, padding = 1)
		self.downsample_conv2d = DownSampleConv2D(input_channels = n_filters ,n_filters=n_filters,kernel_size=kernel_size,padding =1)

		self.resblockdown = nn.Sequential(nn.ReLU(), self.Conv2d, nn.ReLU(), self.downsample_conv2d)

		self.downsample_residual = DownSampleConv2D(
			input_channels=input_channels, n_filters=n_filters, kernel_size=(1,1))

	@torch.jit.script_method
	def forward(self, x):
		# TODO 1.1: Forward through self.layers and implement a residual connection.
		# Make sure to downsample the residual before adding it to the layer output
		x1 = self.downsample_residual(x)
		# x = self.resblockdown(x) + self.downsample_residual(x)
		x2 = self.resblockdown(x)
		return x1 + x2
		# return 0

class ResBlock(torch.jit.ScriptModule):
	# TODO 1.1: Impement Residual Block as described below.
	"""
	ResBlock(
		(layers): Sequential(
			(0): ReLU()
			(1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			(2): ReLU()
			(3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		)
	)
	"""

	def __init__(self, input_channels, kernel_size=(3,3), n_filters=128):
		super(ResBlock, self).__init__()
		# TODO 1.1: Setup the network layers
		self.input_channels = input_channels
		self.kernel_size = kernel_size
		self.n_filters = n_filters

		self.Relu = nn.ReLU()
		self.Conv2d1 = nn.Conv2d(input_channels,n_filters,kernel_size=(3,3), stride =1, padding =1)
		self.Conv2d2 = nn.Conv2d(n_filters,n_filters,kernel_size=(3,3), stride =1, padding =1)

		self.resblock = nn.Sequential(nn.ReLU(), self.Conv2d1,nn.ReLU(), self.Conv2d2)

	@torch.jit.script_method
	def forward(self, x):
		# TODO 1.1: Forward the conv layers. Don't forget the residual connection!
		output = self.resblock(x)


		return output + x

class Generator(torch.jit.ScriptModule):
	# TODO 1.1: Impement Generator. Follow the architecture described below:
	"""
	Generator(
	(dense): Linear(in_features=128, out_features=2048, bias=True)
	(layers): Sequential(
		(0): ResBlockUp(
		(layers): Sequential(
			(0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(1): ReLU()
			(2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			(3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(4): ReLU()
			(5): UpSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			)
		)
		(upsample_residual): UpSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
		)
		)
		(1): ResBlockUp(
		(layers): Sequential(
			(0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(1): ReLU()
			(2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			(3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(4): ReLU()
			(5): UpSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			)
		)
		(upsample_residual): UpSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
		)
		)
		(2): ResBlockUp(
		(layers): Sequential(
			(0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(1): ReLU()
			(2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			(3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
			(4): ReLU()
			(5): UpSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			)
		)
		(upsample_residual): UpSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
		)
		)
		(3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		(4): ReLU()
		(5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		(6): Tanh()
	)
	)
	"""

	def __init__(self, starting_image_size=4):
		super(Generator, self).__init__()
		# TODO 1.1: Setup the network layers
		self.starting_image_size = starting_image_size
		self.n_filters = 128
		self.dense_init = nn.Linear(self.n_filters, 2048)
		self.gen_block = nn.Sequential(
			ResBlockUp(input_channels=self.n_filters, n_filters=self.n_filters, kernel_size=(3,3)),
			ResBlockUp(input_channels=self.n_filters, n_filters=self.n_filters, kernel_size=(3,3)),
			ResBlockUp(input_channels=self.n_filters, n_filters=self.n_filters, kernel_size=(3,3)),
			nn.BatchNorm2d(self.n_filters),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.n_filters, out_channels=3, kernel_size=(3,3), padding=1,stride=1),
			nn.Tanh()
		)
		

	@torch.jit.script_method
	def forward_given_samples(self, z):
		# TODO 1.1: forward the generator assuming a set of samples z have been passed in.
		# Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
		z = self.dense_init(z)
		z = z.reshape(-1, self.n_filters, 4, 4)
		z = self.gen_block(z)
		return z
		# return 0

	@torch.jit.script_method
	def forward(self, n_samples: int = 1024):
		# TODO 1.1: Generate n_samples latents and forward through the network.
		z = torch.randn(n_samples, self.n_filters).cuda()
		out = self.dense_init(z).cuda()
		out = out.reshape(-1, self.n_filters, 4, 4)
		out = self.gen_block(out)
		return out
		# return 0

class Discriminator(torch.jit.ScriptModule):
	# TODO 1.1: Impement Discriminator. Follow the architecture described below:
	"""
	Discriminator(
	(layers): Sequential(
		(0): ResBlockDown(
		(layers): Sequential(
			(0): ReLU()
			(1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			(2): ReLU()
			(3): DownSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			)
		)
		(downsample_residual): DownSampleConv2D(
			(conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
		)
		)
		(1): ResBlockDown(
		(layers): Sequential(
			(0): ReLU()
			(1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			(2): ReLU()
			(3): DownSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			)
		)
		(downsample_residual): DownSampleConv2D(
			(conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
		)
		)
		(2): ResBlock(
		(layers): Sequential(
			(0): ReLU()
			(1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			(2): ReLU()
			(3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		)
		)
		(3): ResBlock(
		(layers): Sequential(
			(0): ReLU()
			(1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			(2): ReLU()
			(3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		)
		)
		(4): ReLU()
	)
	(dense): Linear(in_features=128, out_features=1, bias=True)
	)
	"""

	def __init__(self):
		super(Discriminator, self).__init__()
		# TODO 1.1: Setup the network layers
		n_filters = 128
		self.discriminator = nn.Sequential(
			ResBlockDown(input_channels=3, kernel_size=(3,3), n_filters=n_filters),
			ResBlockDown(input_channels=n_filters, n_filters=n_filters, kernel_size=(3,3)),
			ResBlock(input_channels=n_filters, n_filters=n_filters, kernel_size=(3,3)),
			ResBlock(input_channels=n_filters, n_filters=n_filters, kernel_size=(3,3)),
			nn.ReLU(),
		)
		self.fc = torch.nn.Linear(in_features=n_filters, out_features=1)


	@torch.jit.script_method
	def forward(self, x):
		# TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
		# Make sure to sum across the image dimensions after passing x through self.layers.
		out = self.discriminator(x)
		y = self.fc(torch.sum(out,dim=[2,3]))
		return y
		# return 0



			# 	# discriminator_loss += disc_loss_fn(disc_output_train_batch,disc_output_gen_data,discriminator_output_interpolated)
			# 	discriminator_loss = disc_loss_fn(disc_output_train_batch,disc_output_gen_data)

			# optim_discriminator.zero_gr