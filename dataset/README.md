# Dataset Description
The dataset used in the experiments  (`D.1`, `D.2`, `U`) can be found here: `LINK`.

After downloading, place the datasets in the `dataset` folder in the project root directory, at the same level as the `src` folder.
The train and test partitions are generated from the datasets by executing `s1_preprocessing.py`.

## Data Preprocessing (`s1_preprocessing.py`)

* `s1_preprocessing.py`: generates the training and test partitions in `dataset/train/` and `dataset/test/`. The script performs the following operations:
  - **Concatenation**: Merges the drained datasets into a unified dataset (`D_drained.csv`);
  - **Target-Specific Split**: Generates datasets for each targets (e.g., removing zero-value initial water tables for specific datasets) and splits them into an 80% training set and a 20% testing set (`TEST_SIZE = 0.20`).


## Generated Datasets

* ### Factor of Safety and Depth of the slip surface
  * `train/drained_fos.csv` and `test/drained_fos.csv`
  * `train/undrained_fos.csv` and `test/undrained_fos.csv`\
	These datasets contain the samples used for model development to predict the **Factor of Safety** (`FoS`) and the **depth of the slip surface** (`z_s`) under drained and undrained conditions, respectively.

* ### Final upstream position of the water table
  * `train/drained_zwu.csv` and `test/drained_zwu.csv`
  * `train/undrained_zwu.csv` and `test/undrained_zwu.csv`\
    These datasets contain the samples used for model development to predict the **final upstream position of the water table** (`z_wu^final`) under drained and undrained conditions, respectively.\
    Samples with an initial upstream position of the water table equal to zero (`z_wu^init = 0`) are removed during preprocessing.

* ### Final downstream position of the water table
  * `train/drained_zwd.csv` and `test/drained_zwd.csv`
  * `train/undrained_zwd.csv` and `test/undrained_zwd.csv`\
	These datasets contain the samples used for model development to predict the **final downstream position of the water table** (`z_wd^final`) under drained and undrained conditions, respectively.\
	Samples with an initial downstream position of the water table equal to zero (`z_wd^init = 0`) are removed during preprocessing.


## Mitigation measures
* `mitigation measures.xlsx`: This dataset contains the slope stabilization measures grouped into five categories according to their dominant physical mechanism
  and engineering function. Their qualitative effectiveness is reported with respect to the main controlling factors, namely the depth of the potential failure
  surface and groundwater conditions, in support of the effectiveness and applicability matrices.