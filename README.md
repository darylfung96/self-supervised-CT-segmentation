## Self-supervised Deep Learning Model for COVID-19 CT Lung Image Segmentation Highlighting Putative Causal Relationship among Age, Underlying Disease and COVID-19

    Daryl Fung Lerh Xing, Qian Liu, Judah Zammit, Carson Kai-Sang Leung, PingZhao Hu
    
### Getting Started

---

####Process data for self-supervision
Prepare dataset for Self-Supervised learning from ICTCF. You can obtain the processed ICTCF dataset here:
    
    https://mega.nz/file/b5AkQSoa#wk40ZPGB8IUCLQBU0S8dZl1SJBYeU9TtB3EDeBBbKP0
    
Once you have downloaded the ```ictcf-dataset.zip```, move the ictcf-dataset to the directory of Self-supervised-CT-segmentatiobn. Then run 

    python process_ictcf_for_self.py -- input_folder ictcf-dataset --output_folder ./datasets/medseg
    
---

#### Training for self-supervision
After running the processing data for self-supervision, you can start training for self-supervised for 

    
####