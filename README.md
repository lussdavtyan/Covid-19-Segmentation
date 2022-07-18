## Getting started

Clone the project

```bash
  git clone https://github.com/lussdavtyan/Covid-19-Segmentation
```
Download *COVID-19-CT-Seg_20cases.zip*, *Infection_Mask.zip* and *Lung_Mask.zip* from https://zenodo.org/record/3757476#.YWG1tRDMLyJ

Go to the project directory

```bash
  cd Covid-19-Segmentation
```

Install dependencies

```bash
  pip3 install -r requirements.txt
```

To run the code
```bash
python infection_segmentation.py --img_path --lung_mask_path --inf_mask_path
```