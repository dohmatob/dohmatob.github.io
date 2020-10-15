build:
	jekyll build

serve: build
	jekyll serve --incremental

upload: build
	./upload.sh
