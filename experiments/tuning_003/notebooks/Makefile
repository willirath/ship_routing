# Execute Jupyter notebooks in numerical order

NOTEBOOKS := $(sort $(wildcard [0-9]*.ipynb))

.PHONY: all
all: $(NOTEBOOKS)
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb..."; \
		jupyter nbconvert --execute --to notebook --inplace "$$nb"; \
	done

.PHONY: clean
clean:
	@for nb in $(NOTEBOOKS); do \
		jupyter nbconvert --clear-output --inplace "$$nb"; \
	done
