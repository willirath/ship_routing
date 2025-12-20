# Execute Jupyter notebooks in numerical order

NOTEBOOKS := $(sort $(wildcard [0-9]*.ipynb))

.PHONY: all
all: $(NOTEBOOKS)
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb..."; \
		papermill "$$nb" "executed/$$nb"; \
	done

.PHONY: clean
clean:
	@for nb in $(NOTEBOOKS); do \
		jupyter nbconvert --clear-output --inplace "$$nb"; \
		rm -fv "executed/$$nb"; \
	done
