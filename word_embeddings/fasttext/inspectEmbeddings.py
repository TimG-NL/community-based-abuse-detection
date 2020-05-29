import fasttext
import sys




def inspect_embeddings(filename):
	model = fasttext.load_model(filename)

	wordlist = ['black', 'immigrants', 'gay', 'women', 'fat', 'niggers', 
				'mexicans', 'terrorists', 'arab', 'jews', 'Trump', 'democrats', 
				'republicans', 'climate']
	for word in wordlist:
		print(model.get_nearest_neighbors(word))


def main(argv):
	# Load parameters
	args = argv[1].strip('\n').split('-')
	minCount, ngrams, dims, dataset = int(args[0]), int(args[1]), int(args[2]), args[3]
	print("minCount: {}, ngrams: {}, dims: {}, dataset: {}".format(minCount, ngrams, dims, dataset))

	# Get embedding filenames
	directory = "/data/s2769670/scriptie/embeddings/"
	filename_path = "{}embeddings_{}_{}_{}_{}.model".format(directory, dataset, minCount, ngrams, dims)

	# Inspect embeddings with predefined wordlist
	inspect_embeddings(filename_path)

if __name__ == '__main__':
	main(sys.argv)