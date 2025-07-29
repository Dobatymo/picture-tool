# viewer-gui

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


## missing features

- more raw formats (like ORF, PEF, RAF, SRW, X3F)
- create "view" of folders: ie. load files using path patterns like "pics/2023 */*.jpg"
- fullscreen mode
- more exif info shown (maybe in extra dialog)
- different image sorts (filename, modification date, ...)
- configurable resizer
- improved zoom controls / size modes: fixed scaling
- rotate by changing exif meta only, uses lossless jpeg rotation, don't allow lossy rotations
- show more controls, infos related to to multi-frame images (show 'i / n' pics info, enable preload, jump to first/last, ...)

## missing features (low priority)
- slideshow

## probably wont implement
- transition effects for slideshows
- printing, scanning, device imports

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

# find-dups

## missing features

- folder modes: multiple folder inputs, combine them or treat all of them separately. two folder mode: find dups from first folder in second, but not within the same folder
- ignore files based on meta data, ie. filesize, resolution, ...

## alternatives

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


# compare-gui

## missing features

- show visual diff to reference
- mass update metadata overwrite (similar to prioritize window)
- advanced meta data like quality scores

# batch-edit / batch-edit-gui

# browser-gui

## missing features

- integrate with viewer-gui
