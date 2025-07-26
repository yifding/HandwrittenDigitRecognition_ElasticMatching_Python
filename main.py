import os
import random
import numpy as np
from PIL import Image
from time import gmtime, strftime
from lib.cmdparser import parser
import lib.utils as utils
from tqdm import tqdm

# import bob.ip.gabor.Transform as Transform
# define alternative function to replace it
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

class Transform:
    def __init__(self, number_of_scales=5, number_of_directions=8,
                 sigma=1.0, k_max=np.pi/2, k_fac=2.0):
        self.scales = number_of_scales
        self.directions = number_of_directions
        self.sigma = sigma
        self.k_max = k_max
        self.k_fac = k_fac
        
        # Pre-compute all kernels in a vectorized way
        scales = np.arange(self.scales)
        directions = np.arange(self.directions)
        
        # Create frequency and theta arrays
        freqs = self.k_max / (self.k_fac ** scales)
        thetas = directions * np.pi / self.directions
        
        # Generate all combinations of frequencies and thetas
        self.kernels = []
        for freq in freqs:
            for theta in thetas:
                self.kernels.append(
                    gabor_kernel(frequency=freq, theta=theta,
                                 sigma_x=self.sigma, sigma_y=self.sigma)
                )
                
        self.number_of_wavelets = len(self.kernels)

    def transform(self, image):
        """Transform an image using pre-computed Gabor kernels"""
        image = image.astype(float)
        H, W = image.shape
        
        # Pre-allocate output array for better performance
        out = np.zeros((self.number_of_wavelets, H, W), dtype=np.complex128)
        
        # Apply all kernels (can be parallelized with multiprocessing if needed)
        for i, kern in enumerate(self.kernels):
            # Perform convolution in one step
            real_kern = np.real(kern)
            imag_kern = np.imag(kern)
            
            real_part = ndi.convolve(image, real_kern, mode='reflect')
            imag_part = ndi.convolve(image, imag_kern, mode='reflect')
            out[i] = real_part + 1j * imag_part
            
        return out


def match_n_cost(image_mask_label_array, image_transformed, no_wavelets, resized_size, match_ite, relative_weight):
	"""
	This method causes the graph on an image from the evaluation set to deform elastically to minimize cost
	when matched against the stored graphs from the masking ('train') dataset
	:param image_mask_label_array: image masks from mask ('train') dataset with labels
	:param image_transformed: image from the evaluation transformation after applying the Gabor stack 
	:param resized_size: image size the data have been resized to
	:param match_ite: the number of iterations allowed for matching to a particular mask
	:param relative_weight: relative weight of the two costs (the cost from matching the vertex and the 
							cost from deforming the edges)
	"""

	# Vectorized creation of initial positions
	rows = np.repeat(np.arange(10) * 2, 10)
	cols = np.tile(np.arange(10) * 2, 10)
	initial_positions = list(zip(rows, cols))
	initial_positions_array = np.array(initial_positions)

	min_cost_global = image_mask_index_min_cost_global = 0.

	# iterating through different masks stored from the mask set
	for image_mask_count in range(len(image_mask_label_array)):
		print('Matching with mask no.:{}'.format(image_mask_count))
		image_mask = image_mask_label_array[image_mask_count]['mask']
		image_mask_absolute = np.absolute(image_mask)
		min_cost_local_image_mask_count = float('inf')
		# running match for 'match_ite' iterations
		for ite in range(match_ite):
			if ite > 0:
				# Start with a copy of initial positions array
				positions_array = initial_positions_array.copy()
				
				# Generate random directions for all positions at once (1-4)
				directions = np.random.randint(1, 5, size=len(initial_positions))
				
				# Create masks for each direction
				dir1_mask = (directions == 1)
				dir2_mask = (directions == 2)
				dir3_mask = (directions == 3)
				dir4_mask = (directions == 4)
				
				# Update positions based on direction
				# Direction 1: x+1
				if np.any(dir1_mask):
					positions_array[dir1_mask, 0] += 1
					# Clip to boundaries
					positions_array[dir1_mask, 0] = np.minimum(positions_array[dir1_mask, 0], resized_size - 1)
				
				# Direction 2: x-1
				if np.any(dir2_mask):
					positions_array[dir2_mask, 0] -= 1
					# Clip to boundaries
					positions_array[dir2_mask, 0] = np.maximum(positions_array[dir2_mask, 0], 0)
				
				# Direction 3: y+1
				if np.any(dir3_mask):
					positions_array[dir3_mask, 1] += 1
					# Clip to boundaries
					positions_array[dir3_mask, 1] = np.minimum(positions_array[dir3_mask, 1], resized_size - 1)
				
				# Direction 4: y-1
				if np.any(dir4_mask):
					positions_array[dir4_mask, 1] -= 1
					# Clip to boundaries
					positions_array[dir4_mask, 1] = np.maximum(positions_array[dir4_mask, 1], 0)
				
				# Convert back to list of tuples
				new_positions = [tuple(pos) for pos in positions_array]
				
				# Also update the numpy array for further operations
				new_positions_array = positions_array
			else:
				new_positions = initial_positions.copy()
				new_positions_array = initial_positions_array.copy()

			# Create empty array for local movement
			image_mask_local_movement = np.empty((no_wavelets, resized_size//2, resized_size//2), 'complex128')
			
			# Use the already created positions array
			x_positions = new_positions_array[:, 0]
			y_positions = new_positions_array[:, 1]
			
			# Create grid indices
			grid_size = resized_size // 2
			grid_rows = np.repeat(np.arange(grid_size), grid_size)
			grid_cols = np.tile(np.arange(grid_size), grid_size)
			
			# Assign values in one vectorized operation
			for w in range(no_wavelets):
				image_mask_local_movement[w, grid_rows, grid_cols] = image_transformed[w, x_positions, y_positions]

			# Use the already created positions array
			positions_array = new_positions_array  # Use the numpy array we already created
			positions_count = len(positions_array)
			grid_size = resized_size // 2
			sum_all_edges = 0.0
			
			# Pre-compute all indices for edge connections
			# Vertical connections (top neighbor)
			top_indices = np.arange(positions_count) - grid_size
			valid_top = (top_indices >= 0)
			
			# Vertical connections (bottom neighbor)
			bottom_indices = np.arange(positions_count) + grid_size
			valid_bottom = (bottom_indices < positions_count)
			
			# Horizontal connections (left neighbor)
			left_indices = np.arange(positions_count) - 1
			# Create a mask for valid left indices first
			valid_left_mask = (left_indices >= 0)
			valid_left = np.zeros(positions_count, dtype=bool)
			
			# Only check positions where left_indices are valid
			if np.any(valid_left_mask):
				valid_left_indices = left_indices[valid_left_mask]
				# Compare x-coordinates of current positions with their left neighbors
				valid_left[valid_left_mask] = positions_array[valid_left_indices, 0] < positions_array[valid_left_mask, 0]
			
			# Horizontal connections (right neighbor)
			right_indices = np.arange(positions_count) + 1
			# Create a mask for valid right indices first
			valid_right_mask = (right_indices < positions_count)
			valid_right = np.zeros(positions_count, dtype=bool)
			
			# Only check positions where right_indices are valid
			if np.any(valid_right_mask):
				valid_right_indices = right_indices[valid_right_mask]
				# Compare x-coordinates of current positions with their right neighbors
				valid_right[valid_right_mask] = positions_array[valid_right_indices, 0] > positions_array[valid_right_mask, 0]
			
			# Calculate edge costs safely
			# For top edges
			if np.any(valid_top):
				valid_top_indices = top_indices[valid_top]
				top_edges = positions_array[valid_top] - positions_array[valid_top_indices]
				top_edge_lengths = np.sqrt(np.sum(top_edges**2, axis=1))
				sum_all_edges += np.sum((top_edge_lengths - 1)**2)
				
			# For bottom edges
			if np.any(valid_bottom):
				valid_bottom_indices = bottom_indices[valid_bottom]
				bottom_edges = positions_array[valid_bottom] - positions_array[valid_bottom_indices]
				bottom_edge_lengths = np.sqrt(np.sum(bottom_edges**2, axis=1))
				sum_all_edges += np.sum((bottom_edge_lengths - 1)**2)
				
			# For left edges
			if np.any(valid_left):
				valid_left_indices = left_indices[valid_left]
				left_edges = positions_array[valid_left] - positions_array[valid_left_indices]
				left_edge_lengths = np.sqrt(np.sum(left_edges**2, axis=1))
				sum_all_edges += np.sum((left_edge_lengths - 1)**2)
				
			# For right edges
			if np.any(valid_right):
				valid_right_indices = right_indices[valid_right]
				right_edges = positions_array[valid_right] - positions_array[valid_right_indices]
				right_edge_lengths = np.sqrt(np.sum(right_edges**2, axis=1))
				sum_all_edges += np.sum((right_edge_lengths - 1)**2)

			image_mask_local_movement_real = np.real(image_mask_local_movement)
			image_mask_local_movement_imgn = np.imag(image_mask_local_movement)
			image_mask_local_movement_abs = np.absolute(image_mask_local_movement)

			# Vectorized computation of dot product between mask and local movement
			# Reshape to 2D arrays for easier dot product calculation
			mask_flat = image_mask_absolute.reshape(resized_size//2, resized_size//2, no_wavelets)
			local_flat = image_mask_local_movement_abs.reshape(resized_size//2, resized_size//2, no_wavelets)
			
			# Compute dot products for all vertices at once
			dot_products = np.sum(mask_flat * local_flat, axis=2)
			
			# Compute norms for normalization
			mask_norms = np.sqrt(np.sum(mask_flat**2, axis=2))
			local_norms = np.sqrt(np.sum(local_flat**2, axis=2))
			
			# Avoid division by zero
			mask_norms = np.maximum(mask_norms, 1e-10)
			local_norms = np.maximum(local_norms, 1e-10)
			
			# Compute normalized dot products (cosine similarity)
			normalized_dot_products = dot_products / (mask_norms * local_norms)
			
			# Sum all vertex costs
			sum_all_vertices = np.sum(normalized_dot_products)
		
			# total cost computed
			cost = relative_weight*sum_all_edges - sum_all_vertices

			# cost for deformed position of particular mask that gives minimum cost
			if(cost < min_cost_local_image_mask_count or ite == 0):
				min_cost_local_image_mask_count = cost

		# cost for mask that gives minimum cost among all masks
		if(min_cost_local_image_mask_count < min_cost_global or image_mask_count == 0):
			min_cost_global = min_cost_local_image_mask_count
			image_mask_index_min_cost_global = image_mask_count

	return image_mask_label_array[image_mask_index_min_cost_global]['file_path']

def mask(mask_data, no_wavelets, resized_size, hyperparameter_list):
	"""
	This method stores the masks from the mask ('train') set
	:param mask_data: mask/'train' dataset
	:param no_wavelets: number of wavelets in the Gabor stack (number of directions * number of frequencies)
	:param resized_size: image size the data have been resized to
	:param hyperparameter_list: list for different Gabor hyperparameters
	"""

	# directory traversal
	file_list=[]
	for (dirpath, dirnames, filenames) in os.walk(mask_data):
		file_list.extend(filenames)

	image_count = 0
	for image_file in tqdm(file_list):
		image_file_path = os.path.join(mask_data, image_file)
		image = Image.open(image_file_path)
		label = image_file_path[len(image_file_path):len(image_file_path)+1]
		image =np.array(image)

		gabor_wavelets = Transform(number_of_scales=hyperparameter_list[0],
								number_of_directions=hyperparameter_list[1],
								sigma=hyperparameter_list[3],
								k_max=hyperparameter_list[3],
								k_fac=hyperparameter_list[4])
		image_transformed = gabor_wavelets.transform(image)
		
		# Vectorized selection of pixels at stride 2
		# This creates a view of the original array, with every 2nd element
		image_mask = image_transformed[:, 0:resized_size:2, 0:resized_size:2].copy()

		# Create a structured data object to hold heterogeneous data
		current_item = {
			'mask': image_mask,
			'label': label,
			'file_path': os.path.join(mask_data, image_file)
		}
		
		if image_count == 0:
			image_mask_label_array = [current_item]
		else:
			image_mask_label_array.append(current_item)

		image_count += 1

	return image_mask_label_array

def eval(
	eval_data,
	mask_data,
	resized_size,
	image_mask_label_array,
	no_wavelets,
	log,
	hyperparameter_list,
	match_ite,
	relative_weight,
	args
	):
	"""
	This method stores the masks from the mask ('train') set
	:param eval_data: evaluation dataset
	:param mask_data: mask/'train' dataset
	:param resized_size: image size the data have been resized to
	:param image_mask_label_array: image masks from mask ('train') dataset with labels
	:param no_wavelets: number of wavelets in the Gabor stack (number of directions * number of frequencies)
	:param log: log file
	:param hyperparameter_list: list for different Gabor hyperparameters
	:param match_ite: the number of iterations allowed for matching to a particular mask
	"""

	# directory traversal
	file_list=[]
	for (dirpath, dirnames, filenames) in os.walk(eval_data):
		file_list.extend(filenames)

	image_count = 0
	image_count_match = 0

	for image_file in file_list:
		image_file_path = os.path.join(eval_data, image_file)
		image = Image.open(image_file_path)
		label = image_file_path[len(image_file_path):len(image_file_path)+1]
		image =np.array(image)

		gabor_wavelets = Transform(number_of_scales=hyperparameter_list[0],
   								number_of_directions=hyperparameter_list[1],
   								sigma=hyperparameter_list[3],
   								k_max=hyperparameter_list[3],
   								k_fac=hyperparameter_list[4])
		image_transformed = gabor_wavelets.transform(image)

		match_file_path = match_n_cost(
			image_mask_label_array,
			image_transformed,
			no_wavelets,
			resized_size,
			match_ite,
			relative_weight
		)
		match_label = match_file_path[len(match_file_path):len(match_file_path)+1]

		if match_label == label:
			image_count_match += 1
			image_count += 1
		else:
			image_count += 1
			print("Iteration No.:{}".format(image_count))
			print("Eval Image File:{}".format(image_file_path))
			print("Wrong Match Image File:{}".format(match_file_path))
			log.write("Iteration No.:{}".format(image_count))
			log.write("Eval Image File:{}".format(image_file_path))
			log.write("Wrong Match Image File:{}".format(match_file_path))
		if image_count%int(args.print_freq) == 0:
			print("Iteration No.:{}".format(image_count))
			print("Current Evaluation Accuracy:{}".format((image_count_match*100.)/image_count))
			log.write("Iteration No.:{}".format(image_count))
			log.write("Current Evaluation Accuracy:{}".format((image_count_match*100.)/image_count))

	return (image_count_match*100.)/image_count

def test_with_limited_images(
	eval_data,
	mask_data,
	resized_size,
	no_wavelets,
	hyperparameter_list,
	match_ite,
	relative_weight,
	args,
	limit_mask=1000,
	limit_eval=100
):
	"""
	A test function that processes only a limited number of images
	:param eval_data: evaluation dataset path
	:param mask_data: mask/'train' dataset path
	:param resized_size: image size the data has been resized to
	:param no_wavelets: number of wavelets in the Gabor stack
	:param hyperparameter_list: list of Gabor hyperparameters
	:param match_ite: number of iterations for matching
	:param relative_weight: weight for edge deformation cost vs vertex matching cost
	:param args: command line arguments
	:param limit_mask: maximum number of mask images to process (default: 1000)
	:param limit_eval: maximum number of evaluation images to process (default: 100)
	:return: evaluation accuracy
	"""
	print(f"Running quick test with {limit_mask} mask images and {limit_eval} evaluation images...")
	
	# Create a temporary log file
	test_log_path = os.path.join('./runs', 'test_' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
	if not os.path.exists(test_log_path):
		os.makedirs(test_log_path)
	test_log_file = os.path.join(test_log_path, "test_log")
	test_log = open(test_log_file, "a")
	
	# Get limited mask images
	file_list_mask = []
	for (dirpath, dirnames, filenames) in os.walk(mask_data):
		file_list_mask.extend(filenames[:limit_mask])
		break  # Only process top-level directory
	
	# Process limited mask images
	image_count = 0
	image_mask_label_array = []
	
	for image_file in tqdm(file_list_mask):
		image_file_path = os.path.join(mask_data, image_file)
		image = Image.open(image_file_path)
		label = image_file_path[len(image_file_path):len(image_file_path)+1]
		image = np.array(image)

		gabor_wavelets = Transform(number_of_scales=hyperparameter_list[0],
								number_of_directions=hyperparameter_list[1],
								sigma=hyperparameter_list[3],
								k_max=hyperparameter_list[3],
								k_fac=hyperparameter_list[4])
		image_transformed = gabor_wavelets.transform(image)
		image_mask = image_transformed[:, 0:resized_size:2, 0:resized_size:2].copy()

		# Create a structured data object
		current_item = {
			'mask': image_mask,
			'label': label,
			'file_path': os.path.join(mask_data, image_file)
		}
		
		image_mask_label_array.append(current_item)
		image_count += 1
	
	# Get limited evaluation images
	file_list_eval = []
	for (dirpath, dirnames, filenames) in os.walk(eval_data):
		file_list_eval.extend(filenames[:limit_eval])
		break  # Only process top-level directory
	
	# Process limited evaluation images
	image_count = 0
	image_count_match = 0

	for image_file in tqdm(file_list_eval):
		image_file_path = os.path.join(eval_data, image_file)
		image = Image.open(image_file_path)
		label = image_file_path[len(image_file_path):len(image_file_path)+1]
		image = np.array(image)

		gabor_wavelets = Transform(number_of_scales=hyperparameter_list[0],
								number_of_directions=hyperparameter_list[1],
								sigma=hyperparameter_list[3],
								k_max=hyperparameter_list[3],
								k_fac=hyperparameter_list[4])
		image_transformed = gabor_wavelets.transform(image)

		match_file_path = match_n_cost(
			image_mask_label_array,
			image_transformed,
			no_wavelets,
			resized_size,
			match_ite,
			relative_weight
		)
		match_label = match_file_path[len(match_file_path):len(match_file_path)+1]

		if match_label == label:
			image_count_match += 1
		
		image_count += 1
		print(f"Test Progress: {image_count}/{len(file_list_eval)}, Current Accuracy: {(image_count_match*100.)/image_count:.2f}%")
		test_log.write(f"Test Progress: {image_count}/{len(file_list_eval)}, Current Accuracy: {(image_count_match*100.)/image_count:.2f}%\n")
	
	test_accuracy = (image_count_match*100.)/image_count
	print(f"Test completed! Final test accuracy: {test_accuracy:.2f}%")
	test_log.write(f"Test completed! Final test accuracy: {test_accuracy:.2f}%\n")
	
	test_log.close()
	return test_accuracy

def main():
	save_path = './runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	log_file = os.path.join(save_path, "stdout")
	log = open(log_file, "a")

	args = parser.parse_args()
	print("Command line options:")
	for arg in vars(args):
		print(arg, getattr(args, arg))
		log.write(arg + ':' + str(getattr(args, arg)) + '\n')
		
	# Check if quick test mode is enabled
	quick_test = getattr(args, 'quick_test', False)

	# utils.extract(args.dataset, args.raw_data)
	# utils.resize(args.mask_data, args.resized_size)
	# utils.resize(args.eval_data, args.resized_size)
	# utils.deskew(args.mask_data)
	# utils.deskew(args.eval_data)

	hyperparameter_list = [args.no_scales, args.no_directions, args.sigma,\
						 args.frequency_max, args.frequency_factor]
						 
	if hasattr(args, 'quick_test') and args.quick_test:
		# Run the quick test with limited number of images
		print("Running quick test mode with limited images...")
		eval_accuracy = test_with_limited_images(
			args.eval_data, 
			args.mask_data, 
			args.resized_size, 
			args.no_scales*args.no_directions,
			hyperparameter_list, 
			args.match_ite, 
			args.relative_weight, 
			args,
			args.mask_limit,
			args.eval_limit
		)
	else:
		# Run the full evaluation
		print("Running full evaluation with all images...")
		image_mask_label_array = mask(args.mask_data, args.no_scales*args.no_directions, \
									args.resized_size, hyperparameter_list)
		eval_accuracy = eval(args.eval_data, args.mask_data, args.resized_size, image_mask_label_array, \
							args.no_scales*args.no_directions, log, hyperparameter_list, args.match_ite, \
							args.relative_weight, args)

	print("Final Evaluation Accuracy:{:.3f}".format(eval_accuracy))
	log.write("Final Evaluation Accuracy:{:.3f}".format(eval_accuracy))

	log.close()


if __name__ == '__main__':
	main()
