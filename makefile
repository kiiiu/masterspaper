FILENAME = TeXpaper
PACKAGE_MANAGER_INSTALL = apt-get install
PDFLATEX_OPTS= -synctex=1 -interaction=nonstopmode --shell-escape --enable-write18 -halt-on-error
MODULES = for dir in $(wildcard modules/*); do $(MAKE) $(1) -C $$dir; done

CWD= $(shell pwd)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    INKSCAPE=inkscape
    PDFLATEX=pdflatex $(PDFLATEX_OPTS)
    OPEN_PDF_UTIL=evince
endif
ifeq ($(UNAME_S),Darwin)
    INKSCAPE=/Applications/Inkscape.app/Contents/Resources/script
    PDFLATEX=/usr/local/texlive//2015/bin/universal-darwin/pdflatex
    OPEN_PDF_UTIL=open
endif

IN := $(FILENAME).tex
OUT := $(FILENAME).pdf

GRP=$(CWD)/graphics

SVGS = $(wildcard $(GRP)/*.svg) \
		$(wildcard $(GRP)/combined_img/*.svg) \
		$(wildcard $(GRP)/vector_img/*.svg)

SVGS2EPS := ${SVGS:.svg=.eps}

all: pdf
	@echo Success!

%.svg: %.dot
	@echo making dot!
	@dot -Tsvg $< -o $@
	@mv $@ $(GRP)/var/
	
%.svg: %.dotn
	@echo making dot!
	@neato -Tsvg -Gepsilon=.0000001 $< -o $@
	@mv $@ $(GRP)/var/

%.eps : %.svg
	$(INKSCAPE) -D -z $< -E $@ --export-ignore-filters

pdf_tex: $(SVGS2PDF)
	@echo SVGs to PDF_TEX are done!

png: $(SVGS2PNG)
	@echo SVGs to PNG are done!

svgfromdot: $(SVGFROMDOT_DOT) $(SVGFROMDOT_NEATO)
	@echo DOT to SVG are done!

eps: $(SVGS2EPS)
	@echo SVGs to EPS are done!

clean_graphics: 
	rm -fv $(GRP)/combined_img/*.eps \
		$(GRP)/combined_img/*.pdf* \
		$(GRP)/vector_img/*.eps \
		$(GRP)/vector_img/*.pdf*

_graphics:  eps

graphics: svgfromdot
	@echo Graphics is prepared!
	make _graphics

# for a first time
$(OUT): graphics
	$(PDFLATEX) $(IN)

pdf: $(OUT)
	$(PDFLATEX) $(IN)

update: 
	git pull origin master
	
deps: update
	sudo $(PACKAGE_MANAGER_INSTALL) texlive-full
	sudo $(PACKAGE_MANAGER_INSTALL) texlive-latex-extra

clean:  clean_graphics
	rm -fv *.out *.toc *.log *.aux $FILENAME.pdf* *.tmp *.sta \
		*.nlo *.idx *.ilg *.nls *.synctex.gz

rebuild: clean all

test: all
	$(OPEN_PDF_UTIL) $(FILENAME).pdf
	
