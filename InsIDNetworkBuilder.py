from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Input, concatenate, Concatenate, Reshape, Flatten, Dense
from AttentionAugmentedConvLayer import augmented_conv2d

###
###Function to create a deep CNN augmented with attention. parameters are defined as follows
###inputShape - shape of the input for the network
###filterSize - size of the filters for convolution operations
###blocks - total number of convolution blocks
###attnBlocks - number of convolution blocks to augment with attention. Note: this will occur in reverse order (i.e. the last blocks will be augmented first)
###attnAmt - the percent of filters (expressed as a decimal) to be set for attention in the augmented blocks.
###headCount - number of attention heads to use
###initializerType - type of initialization to use for the weights and biases
###outputSize - number of items to classify between
###
###Returns: model, modelString
###model - tensorflow model created using the functional approach 
###modelString - string representation of the model including some of the specified parameters 
def createInsIDAttnCNN(inputShape, filterSize=(5, 5), blocks=5, attnBlocks=2, attnAmt=0.25, headCount=8, initializerType='random_normal', outputSize=19):
	
	if(attnBlocks > blocks):
		print("You cannot have more attention blocks than total blocks!! Returning None, None")
		Assert(attnBlocks <= blocks)
		return None, None
	
	inputs = keras.Input(shape=inputShape)
	
	if(attnBlocks == blocks):
		x = augmented_conv2d(inputs, filters=32, kernel_size=filterSize, depth_k=attnAmt, depth_v=attnAmt, num_heads=headCount, relative_encodings=True)#relative_encodings=True
		x = layers.Activation("relu")(x)
		x = augmented_conv2d(x, filters=32, kernel_size=filterSize, depth_k=attnAmt, depth_v=attnAmt, num_heads=headCount, relative_encodings=True)
		x = layers.Activation("relu")(x)
		x = MaxPooling2D((2,2), padding='same')(x)
	else:
		x = augmented_conv2d(inputs, filters=32, kernel_size=filterSize, depth_k=0, depth_v=0, num_heads=8, relative_encodings=True)#relative_encodings=True
		x = layers.Activation("relu")(x)
		x = augmented_conv2d(x, filters=32, kernel_size=filterSize, depth_k=0, depth_v=0, num_heads=8, relative_encodings=True)
		x = layers.Activation("relu")(x)
		x = MaxPooling2D((2,2), padding='same')(x)
	
	left = blocks-1
	for i in range(left):
		if(left - i <= attnBlocks):
			x = augmented_conv2d(x, filters=(64 * (2**i)), kernel_size=filterSize, depth_k=attnAmt, depth_v=attnAmt, num_heads=headCount, relative_encodings=True)
			x = layers.Activation("relu")(x)
			x = augmented_conv2d(x, filters=(64 * (2**i)), kernel_size=filterSize, depth_k=attnAmt, depth_v=attnAmt, num_heads=headCount, relative_encodings=True)
			x = layers.Activation("relu")(x)
			x = MaxPooling2D((2,2), padding='same')(x)
		else:
			x = augmented_conv2d(x, filters=(64 * (2**i)), kernel_size=filterSize, depth_k=0, depth_v=0, num_heads=8, relative_encodings=True)
			x = layers.Activation("relu")(x)
			x = augmented_conv2d(x, filters=(64 * (2**i)), kernel_size=filterSize, depth_k=0, depth_v=0, num_heads=8, relative_encodings=True)
			x = layers.Activation("relu")(x)
			x = MaxPooling2D((2,2), padding='same')(x)
	
	x = Flatten()(x)
	
	#final dense layers to perform classification
	x = Dense(1024, activation='relu', kernel_initializer=initializerType)(x)
	x = Dense(512, activation='relu', kernel_initializer=initializerType)(x)
	x = Dense(256, activation='relu', kernel_initializer=initializerType)(x)
	x = Dense(128, activation='relu', kernel_initializer=initializerType)(x)
	x = Dense(64, activation='relu', kernel_initializer=initializerType)(x)
	outputs = Dense(outputSize, kernel_initializer=initializerType, activation='softmax')(x) 

	model = Model(inputs, outputs)

	modelString = "InsIDAttnCNN_" + str(blocks) + "Blocks_" + str(attnBlocks) + "AttnBlcks_" + str(attnAmt) + "Attn_" + str(headCount) + "Heads"
	return model, modelString