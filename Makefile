build:
	jekyll build

serve: build
	jekyll serve

upload: build
	./upload.sh
