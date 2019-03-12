
## Data Explanation

For more detailed information, please refer to the [DeepDTA article](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245).

### Similarity files

For each dataset, there are two similarity files, drug-drug and target-target similarities.
*  Drug-drug similarities obtained via Pubchem structure clustering.
*  Target-target similarities are obtained via S-W similarity.

These files were used to re-produce the results of two other methods [(Pahikkala et al., 2017)](https://academic.oup.com/bib/article/16/2/325/246479) and [(He et al., 2017)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z), and also for some experiments in DeepDTA model, please refer to [paper](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245). 
*  The original Davis data and more explanation can be found [here](http://staff.cs.utu.fi/~aatapa/data/DrugTarget/).
*  The original KIBA data and more explanation can be found [here](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z).

### Binding affinity files

*  For davis dataset, standard value is Kd in nM. In the article, we used the transformation below:

<a href="https://www.codecogs.com/eqnedit.php?latex=pK_{d}=-log_{10}\frac{K_d}{1e9}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?pK_{d}=-log_{10}\frac{K_d}{1e9}" title="pK_{d}=-log_{10}\frac{K_d}{1e9}" /></a>

* For KIBA dataset, standard value is KIBA score. Two versions of the binding affinity value txt files correspond the original values and transformed values ([more information here](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z)). In the article we used the tranformed form. 

* nan values indicate there is no experimental value for that drug-target pair.


### Train and test folds
There are two files for each dataset: train fold and test fold. Both of these files keep the position information for the binding affinity value given in binding affinity matrices in the text files. 
*  Since we performed 5-fold cv, each fold file contains five different set of positions.
*  Test set is same for all five training sets.

### For using the folds

```python
import pickle
import numpy as np

Y = pickle.load(open("Y", "rb"))
label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
```




