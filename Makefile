download-dataset:
	mkdir -p tmp
	curl -L -o tmp/inpainting.zip "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/inpainting.zip?download=true"
	curl -L -o tmp/insight.zip "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/insight.zip?download=true"
	curl -L -o tmp/text2img.zip "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/text2img.zip?download=true"
	curl -L -o tmp/wiki.zip "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/wiki.zip?download=true"

unzip-dataset:
	mkdir -p images
	unzip -o tmp/inpainting.zip -d images/inpainting
	unzip -o tmp/insight.zip -d images/insight
	unzip -o tmp/text2img.zip -d images/text2img
	unzip -o tmp/wiki.zip -d images/wiki

