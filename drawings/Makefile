TEXOPTS = --output-directory=build --lualatex

TARGET = test_drawings

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex | build
	latexmk $(TEXOPTS) $(TARGET).tex

build:
	mkdir -p build

clean:
	rm -rf build
