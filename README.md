# panoramas-image-stitching


## Requirement

- python3 (or higher)
- opencv 3.0 (or higher)

You will need to install some package using `pip3`:

- numpy
- matplotlib


## Usage

```bash
$ python main.py <input img dir>

# for example
$ python ./main.py ../input_image/Xue-Mountain-Enterance/
```

## Input format

The input dir should have:

- Some `.png` or `.jpg` images
- A `image_list.txt`, file should contain:
  - filename
  - focal_length

This is an example for `image_list.txt`:

```
# Filename   focal_length
DSC_0184.jpg 830
DSC_0185.jpg 830
DSC_0186.jpg 830
DSC_0187.jpg 830
DSC_0171.jpg 830
DSC_0172.jpg 830
DSC_0173.jpg 830
DSC_0174.jpg 830
DSC_0175.jpg 830
DSC_0176.jpg 830
DSC_0177.jpg 830
DSC_0178.jpg 830
DSC_0179.jpg 830
DSC_0180.jpg 830
DSC_0182.jpg 830
DSC_0183.jpg 830
```


## Output

The program will output:

- Every stitched images, with filename `0.jpg`, `1.jpg`, `2.jpg`, ...
- A aligned image `aligned.jpg`
- A cropped image `cropped.jpg`

## Parameters

The program have some constant parameters that can easily changed in `constant.py`.

## Environment

I test my code in Window10/Linux/MacOS.  
It should work fine in these system.
