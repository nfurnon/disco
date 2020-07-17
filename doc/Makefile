# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIBUILD= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
CODEDIR       = ../disco_theque
APIDOCOUT     = "$(SOURCEDIR)"/apidoc


.PHONY: all help Makefile

all: apidoc html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@echo "Generating API doc"
	@$(SPHINXAPIBUILD) -fTMe "$(CODEDIR)" -o "$(APIDOCOUT)"
