// tr
python nii_to_jpg.py --nii_file InfNet/Dataset/tr_im.nii --filename_prefix tr --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/tr_im.nii --filename_prefix tr --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg

python nii_to_jpg.py --nii_file InfNet/Dataset/tr_mask.nii --filename_prefix tr --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/tr_mask.nii --filename_prefix tr --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary
python nii_to_jpg.py --nii_file InfNet/Dataset/tr_mask.nii --filename_prefix tr --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

// rp
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/1.nii --filename_prefix rp_1 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/2.nii --filename_prefix rp_2 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/3.nii --filename_prefix rp_3 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/4.nii --filename_prefix rp_4 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/5.nii --filename_prefix rp_5 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/6.nii --filename_prefix rp_6 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/7.nii --filename_prefix rp_7 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/8.nii --filename_prefix rp_8 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/9.nii --filename_prefix rp_9 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/Imgs/ --save_type jpg

python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/1.nii --filename_prefix rp_1 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/2.nii --filename_prefix rp_2 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/3.nii --filename_prefix rp_3 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/4.nii --filename_prefix rp_4 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/5.nii --filename_prefix rp_5 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/6.nii --filename_prefix rp_6 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/7.nii --filename_prefix rp_7 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/8.nii --filename_prefix rp_8 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_im/9.nii --filename_prefix rp_9 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Imgs/ --save_type jpg


//rp_1_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/1.nii --filename_prefix rp_1 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/1.nii --filename_prefix rp_1 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/1.nii --filename_prefix rp_1 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_2_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/2.nii --filename_prefix rp_2 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/2.nii --filename_prefix rp_2 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/2.nii --filename_prefix rp_2 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_3_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/3.nii --filename_prefix rp_3 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/3.nii --filename_prefix rp_3 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/3.nii --filename_prefix rp_3 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_4_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/4.nii --filename_prefix rp_4 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/4.nii --filename_prefix rp_4 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/4.nii --filename_prefix rp_4 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_5_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/5.nii --filename_prefix rp_5 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/5.nii --filename_prefix rp_5 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/5.nii --filename_prefix rp_5 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_6_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/6.nii --filename_prefix rp_6 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/6.nii --filename_prefix rp_6 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/6.nii --filename_prefix rp_6 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_7_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/7.nii --filename_prefix rp_7 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/7.nii --filename_prefix rp_7 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/7.nii --filename_prefix rp_7 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_8_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/8.nii --filename_prefix rp_8 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/8.nii --filename_prefix rp_8 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/8.nii --filename_prefix rp_8 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

//rp_9_mask
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/9.nii --filename_prefix rp_9 --output_folder InfNet/Dataset/TrainingSet/LungInfection-Train/GT/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/9.nii --filename_prefix rp_9 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/Prior/ --save_type png --is_binary True
python nii_to_jpg.py --nii_file InfNet/Dataset/rp_msk/9.nii --filename_prefix rp_9 --output_folder InfNet/Dataset/TrainingSet/MultiClassInfection-Train/GT/ --save_type png

