PYINSTALLER_BIN ?= pyinstaller
SPEC ?= powercheck.spec

.PHONY: dist
dist:
	$(PYINSTALLER_BIN) powercheck.spec

.PHONY: clean
clean:
	rm -rf dist build

# helpful for debugging
print-%: ;@echo $*=$($*)
