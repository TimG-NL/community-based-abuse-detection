import fasttext
import sys




def inspect_embeddings(filename):
	model = fasttext.load_model(filename)
	
	wordlist = ['black', 'immigrants', 'gay', 'woman', 'man', 'trans', 'fat', 'niggers', 
				'feminist', 'terrorists', 'arab', 'jews', 'trump', 'democrats', 
				'republicans', 'climate', 'religion', 'god', 'islam', 'christianity']
	word_neighbors = {}
	for word in wordlist:
		word_neighbors[word] = model.get_nearest_neighbors(word)
		
	for word, values in word_neighbors.items():
		print(word)
		[print(value) for value in values]
		print("\n")


def main(argv):
	# Load parameters
	args = argv[1].strip('\n').split('-')
	minCount, ngrams, dims, dataset = int(args[0]), int(args[1]), int(args[2]), args[3]
	print("minCount: {}, ngrams: {}, dims: {}, dataset: {}".format(minCount, ngrams, dims, dataset))

	# Get embedding filenames
	directory = "../../data/embeddings/fasttext/"
	if dataset == 'abusive':
		filename_path = "{}embeddings_{}_{}_{}_{}_large.model".format(directory, dataset, minCount, ngrams, dims)
	else:
		filename_path = "{}embeddings_{}_{}_{}_{}.model".format(directory, dataset, minCount, ngrams, dims)
	# Inspect embeddings with predefined wordlist
	inspect_embeddings(filename_path)

if __name__ == '__main__':
	main(sys.argv)
