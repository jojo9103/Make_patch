# Make_patch
SPlit patch using pathology image

Macenko normalization.py is used for normalization of pathological images.

Wsi tiling analysis.py is used to create patches on a micrometer basis.

wsi_tiling(File,dest_imagePath,img_name,Tile_size,color_norm=False, tumor_mask=None, debug=False,parallel_running=True)

File = WSI file
dest_imagePath = Save Path
Tile_size = pixel based
color_norm = TRUE/FALSE # macenko normalization
tumor_mask = 
debug =
parallel_running = True 
