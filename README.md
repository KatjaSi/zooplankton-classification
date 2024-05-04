# Zooplankton Classification

## Extracting images from .7z file

To extract images from the `.7z` archive:


```bash
python src/data/extract_images.py --zip_file_path=<path_to_zip_file>
```

## Printing statistics

#### class statistics

```bash
python src/data/data_statistics.py --class_name detritus
```

#### category statistics

```bash
python src/data/data_statistics.py --category_name COP
```