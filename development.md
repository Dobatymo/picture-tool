# batch-edit

## implemented features

- Add date stamps to pictures. The date/time is read from the EXIF information.
- resize
- rotate according to exif orientation

# viewer

## goals

1. don't do unexpected things
  - never modify files without explicit user agreement
  - have consistent image rendering quality
  - native UI controls (default shortcuts, common widget functions)
2. performance
  - fast display and switching between images (using caches, pre-loading, ...)
  - fast startup (use persistent process)
3. lossless editing and management, eg.
  - delete by move
  - rotate losslessly or by metadata
  - filters only applied to view (possibly with sidecar files)
4. multi-platform

## implemented features

- supported file formats jpeg, png, heic, dng (and more)
- two-way buffer (going forward and backward to next/previous image)
- rotate for viewing based on exif data
- user defined hotkeys for (some) custom functions
- view filters: grayscale, histogram normalization

## missing features

- create "view" of folders: ie. load files using path patterns like "pics/2023 */*.jpg"
- fullscreen mode
- slideshow
- more exif info shown (maybe in extra dialog)
- different image sorts (filename, modification date, ...)
- configurable resizer
- improved zoom controls / size modes: fixed scaling
- rotate by changing exif meta only, uses lossless jpeg rotation, don't allow lossy rotations

## possible optimizations

- right now there are some unnecessary (?) copies and slow tobytes/frombytes calls which could be improved
- pre-resolve all city locations (or other expensive meta data) for all paths at once in the background
- some memory leaks

## alternatives

- ACDSee Photo Studio (Windows 7+): inconsistent display quality, slow
- Microsoft Photos (Windows 8+): no features whatsoever
- Apple Photos
- Adobe Lightroom Classic
- XnView MP (Windows 7+, Mac 10.13+, Linux)
- ImageGlass: open source, slow

# compare-gui

## implemented features

- show groups of files in table
  - sortable by columns
  - change priority of files (all apart from top priority are checkable)
- image view to quickly switch between dup groups and display basic meta info
- mass prioritization dialog window
  - multiple criteria
  - multiple ranking functions based on user input
- load/safe multiple file formats

## missing features

- show visual diff to reference
- mass update metadata overwrite (similar to prioritize window)
- advanced meta data like quality scores

# browser-gui

## missing features

- integrate with viewer-gui

# find-dups

## implemented features

- dup modes: file hash, perceptual hash, filesize
- multiprocessing
- export results as csv (and include metadata)

## missing features

- folder modes: multiple folder inputs, combine them or treat all of them separately. two folder mode: find dups from first folder in second, but not within the same folder
- ignore files based on meta data, ie. filesize, resolution, ...

### alternatives

- qarmin/czkawka
- arsenetar/dupeguru

- idealo/imagededup. pros: CNN, cons: memory issues with CNN...
- elisemercury/Duplicate-Image-Finder. cons: no multiprocessing
- InexplicableMagic/photodedupe. Rust.
- jesjimher/imgdupes
- rif/imgdup2go
- markusressel/py-image-dedup
- opennota/findimagedupes
- magamig/duplicate-images-finder
- DragonOfMath/dupe-images. node.js.
- knjcode/imgdupes
- beeftornado/duplicate-image-finder: old and not maintained

## resources

- https://rmlint.readthedocs.io/en/latest/cautions.html
