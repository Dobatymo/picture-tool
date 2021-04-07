# picture-tool

- Add date stamps to pictures. The date/time is read from the EXIF information.
- Resize and rotate

## Help

- `py statistics.py --help`
```
usage: statistics.py [-h] [--rebuild-cache] [--verbose] [--cache-path PATH] [--select FEATURE [FEATURE ...]] [--sort-values]
                     [--out PATH] [--groupby FEATURE | --filter FEATURE VALUE | --sql QUERY]
                     path

positional arguments:
  path                  Input directory to scan

optional arguments:
  -h, --help            show this help message and exit
  --rebuild-cache       Forces a cache rebuild
  --verbose             Debug output
  --cache-path PATH     Path to cache
  --select FEATURE [FEATURE ...]
                        Features to include in output
  --sort-values         Sort output by value instead of key
  --out PATH            If given write csv output to file
  --groupby FEATURE     Collect statistics for feature. Allowed features are: 0th Artist, 0th Bits per sample, 0th Compression, 0th
                        Copyright, 0th DateTime, 0th ExifTag, 0th GPSTag, 0th Host computer, 0th Image description, 0th Image length,
                        0th Image width, 0th Make, 0th Model, 0th Orientation, 0th Photometric interpretation, 0th Planar configuration,
                        0th Processing software, 0th Rating, 0th RatingPercent, 0th Resolution unit, 0th Samples per pixel, 0th
                        Software, 0th Tile length, 0th Tile width, 0th YCbCr positioning, 1st Compression, 1st Image length, 1st Image
                        width, 1st JPEGInterchangeFormat, 1st JPEGInterchangeFormatLength, 1st Orientation, 1st Resolution unit, 1st
                        YCbCr positioning, Exif Body serial number, Exif Color space, Exif Contrast, Exif Custom rendered, Exif DateTime
                        digitized, Exif DateTime original, Exif Exposure mode, Exif Exposure program, Exif Flash, Exif Focal length in
                        35mm film, Exif Focal plane resolution unit, Exif Gain control, Exif ISO speed ratings, Exif Image unique ID,
                        Exif Interoperability tag, Exif Lens make, Exif Lense model, Exif Light source, Exif Metering mode, Exif Offset
                        time, Exif Offset time digitized, Exif Offset time original, Exif Pixel X dimension, Exif Pixel Y dimension,
                        Exif Recommended exposure index, Exif Saturation, Exif Scene capture type, Exif Sensing method, Exif Sensitivity
                        type, Exif Sharpness, Exif Sub seconds time, Exif Sub seconds time digitized, Exif Sub seconds time original,
                        Exif Subject area, Exif Subject distance range, Exif White balance, File size, GPS GPS DateStamp, GPS GPS
                        destination bearing reference, GPS GPS geodetic datum, GPS GPS image direction reference, GPS GPS latitude
                        reference, GPS GPS longitude reference, GPS GPS satellites, GPS GPS speed reference, File modification time
  --filter FEATURE VALUE
  --sql QUERY           Query by SQL. Use table `df`.
```

## Examples

- `py statistics.py "C:\Users\Public\Pictures" --groupby "0th Model"`

Show a list of all different camera models found with a count of how often they are used.

- `py statistics.py "C:\Users\Public\Pictures" --filter "0th Model" "iPhone XR" --select path`

Find all pictures taken with an iPhone XR and show the file paths.

- `py statistics.py --sql "SELECT path, `0th Make`, `0th Model` FROM df WHERE `0th Model` LIKE '%canon eos%'" D:\`

Find all pictures taken by a Canon EOS camera using SQL query.
