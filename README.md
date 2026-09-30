# LIBRARY

tensorflow # (membuat neurol network, training, membaca dataset, menyimpan model, convert to .tflite)
numpy # untuk operasi data numerik
pillow # membaca dan proses gambar e.g -> .jpg, .png
matplotlib # membuat grafik e.g -> accuracy.png
scikit-learn # proses machine learning

# PROSES

## Konsep
Dataset -> Mask / No Mask -> Neural Network -> Training -> Model.

## Dataset
Sample untuk model AI agar tahu apa itu masker. 
dari model yang disiapkan, nantinya contoh tersebut digunakan untuk model belajar mencari pola.
Kemudian, nama folder pada dataset akan otomatis menjadi label/class untuk model.
Mapping 0 = mask, 1 = no mask atau sebaliknya tergantung urutan baca tensorflow.

### NEURAL NETWORK (Jaringan saraf tiruan) 
merupakan sistem komputasi dalam kecerdasan buatan (AI) yang dirancang untuk meniru cara kerja otak manusia dalam memproses informasi.


### TRAINING
setelah training atau running train_mask_detector.py,
maka akan mendapatkan "face_mask_model.keras" yang merupakan model TensorFlow/Keras.