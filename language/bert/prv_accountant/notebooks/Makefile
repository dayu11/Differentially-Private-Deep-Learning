
.PHONY: all
all:
	find *.ipynb -exec jupyter nbconvert --to notebook --inplace --execute {} +


.PHONY: clean
clean:
	find *.ipynb -exec jupyter nbconvert --clear-output --inplace {} \;
