# 537cls - classify 537 noisemes

Predict from 537 classes of noiseme, as specified in [class_labels_indices.csv](https://github.com/srvk/537cls/blob/master/class_labels_indices.csv)  
Output is in RTTM format, tab separated. (Labels are quoted and may contain spaces)  
Also outputs the frame probability matrix  

## Dependencies
Requires Pytorch compiled for CPU, not GPU, such as installed as part of http://github.com/srvk/DiViMe

## Installation: Model Download
First download the model to the folder where this repository has been cloned:
```
wget http://speechkitchen.org/model.pt
```

## Quickstart
To run:
```
python predict.py <audiofile>.wav
```

Numerous audio formats are supported. Output in <audiofile>.rttm
### Selftest
  ```
  python predict.py example.mp3
  
  cat example.rttm
  vagrant@vagrant-ubuntu-trusty-64:~/537cls$ cat example.rttm 
SPEAKER example 1       6.4     0.3     <NA>    <NA>    "Speech"        <NA>    <NA>
SPEAKER example 1       6.9     1.3     <NA>    <NA>    "Speech"        <NA>    <NA>
SPEAKER example 1       8.5     1.2     <NA>    <NA>    "Speech"        <NA>    <NA>
SPEAKER example 1       10.1    1.0     <NA>    <NA>    "Speech"        <NA>    <NA>
SPEAKER example 1       0.2     6.1     <NA>    <NA>    "Music" <NA>    <NA>
SPEAKER example 1       8.2     0.4     <NA>    <NA>    "Music" <NA>    <NA>
SPEAKER example 1       9.8     0.3     <NA>    <NA>    "Music" <NA>    <NA>
SPEAKER example 1       11.1    4.0     <NA>    <NA>    "Music" <NA>    <NA>
SPEAKER example 1       15.2    0.1     <NA>    <NA>    "Music" <NA>    <NA>
SPEAKER example 1       0.8     0.5     <NA>    <NA>    "Brass instrument"      <NA>    <NA>
SPEAKER example 1       1.5     0.3     <NA>    <NA>    "Brass instrument"      <NA>    <NA>
SPEAKER example 1       3.0     0.4     <NA>    <NA>    "Brass instrument"      <NA>    <NA>
SPEAKER example 1       4.7     0.1     <NA>    <NA>    "Brass instrument"      <NA>    <NA>
SPEAKER example 1       13.1    0.3     <NA>    <NA>    "Brass instrument"      <NA>    <NA>
SPEAKER example 1       0.9     0.2     <NA>    <NA>    "Trombone"      <NA>    <NA>
SPEAKER example 1       3.0     0.2     <NA>    <NA>    "Inside, large room or hall"    <NA>    <NA>
SPEAKER example 1       4.4     0.3     <NA>    <NA>    "Television"    <NA>    <NA>
SPEAKER example 1       11.5    0.4     <NA>    <NA>    "Television"    <NA>    <NA>
SPEAKER example 1       13.6    0.9     <NA>    <NA>    "Television"    <NA>    <NA>
SPEAKER example 1       14.6    0.1     <NA>    <NA>    "Television"    <NA>    <NA>
  ```
