# picture-tool

This is a collection of several picture/image/photo related software.

## build

- `torch` is only available as wheels for x64 for macOS for `<2.3`. `>=2.3` only has wheels for arm64.
- `scikit-image==0.18.3` (used for Python 3.8 and 3.9) only has x64 wheels available.

Because of that, Python 3.8 and 3.9 is tested with macOS 13 x64 on GitHub.

# scripts

## quality.py

### Setup

Download cv2 models

- `curl https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml -o brisque_range_live.yml`
- `curl https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml -o brisque_model_live.yml`

Install cmake and have cmake in `PATH`.

## Statistics

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

### Examples

- `py statistics.py "C:\Users\Public\Pictures" --groupby "0th Model"`

Show a list of all different camera models found with a count of how often they are used.

- `py statistics.py "C:\Users\Public\Pictures" --filter "0th Model" "iPhone XR" --select path`

Find all pictures taken with an iPhone XR and show the file paths.

- ``py statistics.py --sql "SELECT path, `0th Make`, `0th Model` FROM df WHERE `0th Model` LIKE '%canon eos%'" D:\``

Find all pictures taken by a Canon EOS camera using SQL query.

### Features I would like to have

- support metadata: filesystem, Exif, IPTC, XMP, ...
- support inferred metadata: technical/aesthetic quality score, orientation, persons, setting, objects
- manual meta data: notes, personal-score, photographer name

- GUI/CLI: support SQL syntax for finding and mass updating images and metadata
	- rename files: `UPDATE /images/2022-01-01 SET filename = format("$date ($camera)", date, camera) WHERE camera = NIKON`
	- count pics in folder and order by largest folder: `SELECT count(*) AS num, path FROM / GROUP BY directory SORT BY num DESC`
	- find best pic in every folder: `SELECT max_by(path, quality_score) FROM / GROUP BY directory`
	- find all pics in city: `SELECT path FROM / WHERE is_close(location, "Tokyo")`
	- move all pics of person to subfolder: `UPDATE / SET directory = directory/$person WHERE person = Jack`
	- rotate pics correctly `UPDATE / SET file_orientation = inferred_orientation WHERE file_orientation != inferred_orientation`
	- find duplicates (handwavy) `SELECT path, hamming_distance(A.sem_hash, B.sem_hash) AS dist FROM / AS A JOIN / AS B ON dist < 2 GROUP BY A.path FOREACH ORDER BY resolution`
	- set exif time to mod time if exif time not available: `UPDATE / SET exif_date = mod_time WHERE exif_date is null`
	- standard prune operation: delete to bin, delete, move to dir, copy to dir, replace with (sym/hard-link)
	- remember pruned files and optionally auto-prune new previously pruned files
- GUI: browse image using map based on GPS or other meta data

## find-dups

```
usage: find-dups.py [-h] [--extensions .EXT [.EXT ...]] [-r] [-v] [--mode {file-hash,image-hash,phash}] [--hashdb PATH] [--normalize OP [OP ...]]
                    [--resolution-normalized N N] [--parallel-read N] [--chunksize N] [--out PATH] [--ntfy-topic NTFY_TOPIC] [--overwrite-cache] [--version]
                    DIR [DIR ...]

Find picture duplicates

positional arguments:
  DIR                   Input directories

optional arguments:
  -h, --help            show this help message and exit
  --extensions .EXT [.EXT ...]
                        Image file extensions (default: ('.jpg', '.jpeg', '.heic', '.heif', '.png', '.webp'))
  -r, --recursive       Read directories recursively (default: False)
  -v, --verbose         Verbose output (default: False)
  --mode {file-hash,image-hash,phash}
                        Hashing mode. `file-hash` simply hashes the whole file. `image-hash` hashes the uncompressed image data of the file and normalizes the
                        rotation. `phash` calculates the perceptual hash of the image. (default: image-hash)
  --hashdb PATH         Path to sqlite database file to store hashes. (default: C:\Users\<username>\AppData\Local\Dobatymo\picture-tool\hashes.sqlite)
  --normalize OP [OP ...]
                        Normalization operations. Ie. when orientation is normalized, files with different orientations can be detected as duplicates
                        (default: ('orientation', 'resolution', 'colors'))
  --resolution-normalized N N
                        All pictures will be resized to this resolution prior to comparison. It should be smaller than the smallest picture in one duplicate
                        group. If it's smaller, more differences in image details will be ignored. (default: (256, 256))
  --parallel-read N     Default read concurrency (default: 4)
  --chunksize N         Specifies the number of hashes to compare at the same the time. Larger chunksizes require more memory. (default: 2000)
  --out PATH            Write results to file. Otherwise they are written to stdout. (default: None)
  --ntfy-topic NTFY_TOPIC
                        Get notifications using *ntfy* topics. Useful for long-running scans. (default: dobatymo-xjMiBan0A)
  --overwrite-cache     Update cached values (default: False)
  --version             show program's version number and exit
```
