# Resampling
There are two main strategies for handling the issue of different kinematic properties for different jet flavours. These are reweighting and resampling. Reweighting applies a weight to each data point when calculating the loss, resulting in a weighted average loss for the considered data points. Resampling changes the actual data distribution by over- or undersampling data to achieve the desired data distribution. Statistically, the two methods are equivalent (in expectation). However, [in some cases](https://web.stanford.edu/~lexing/resw.pdf), resampling may result in a more numerically stable approximation of the minima of the loss function. Empirically, it has been found that resampling produces better results for jet flavour tagging performance.

For resampling, UPP has two different methods implemented. The desired method (`pdf` or `countup`) can be specified in the configuration file.

### Skipping resampling

Resampling can be disabled entirely by either omitting the `resampling` block from the config or setting `method: none`. In this case no `target`, resampling `variables`, or histogram (`--prep`) step are required. The objects passing the cuts are written directly, capped at each component's `num_global_objects`. Setting `num_global_objects: -1` (also valid for `num_global_objects_val` / `num_global_objects_test`) writes **all** objects of that component passing the cuts.

Note that the `--no-resample` command line flag is different: it only skips the resampling *stage* (for example to re-run the merge/norm/plot stages on existing component files) and does not disable resampling.

### PDF (probability density function)

This is an implementation of an [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling) method. The aim of the algorithm is to ensure that the kinematic probability density functions (pdfs) of all classes are matched to that of a chosen target class.

The resampling is done using the following steps:

1. A `num_global_objects_estimate_hist` number of objects are binned for each class using the configurations for resampling variable bins. This histogram, `pdf_resampled_flavour`, is the initial estimate of the pdf of the objects of each class.
2. The importance function is estimated by using the ratio of the histograms for each class to that of the target class, `pdf_target_flavour/pdf_resampled_flavour`. Safe division is used, which ensures that if for a bin in `pdf_resampled_flavour` is 0, we skip that bin. This ensures that we do not divide by 0. If a bin in `pdf_target_flavour` is 0, we also skip the bin. 
3. Optionally, the importance function is upscaled. This means that it is interpolated using cubic spline interpolation to a finer grid of bins. The centres of bins are used as nodes for the splines. The new bins are created by splitting the old bins into `upscale_pdf` number of bins of equal width. The function is evaluated in the centers of the new bins. This way, the edge bins of each binning region are actually extrapolated rather than interpolated.
4. The new batch of objects is being read and after the cuts are applied `n_batch` objects remain. The objects are binned with the binning from step 1 (if upscaling is not used) or the upscaled binning defined by 3 (if upscaling is used) and the reference number of the bin for each object is saved.
5. Each object is assigned an importance score equal to the value of the importance function in the corresponding bin.
6. `n_batch*flavour.sampling_fraction` objects are selected with replacement using importance scores as weights.

This algorithm is used for all the classes except the target class, for which all objects are saved without sampling as they already follow the desired distribution. One has to remember that `flavour.sampling_fraction==1` will lead to many objects being selected more then once, choosing lower `sampling_fractions` can help against it.

### Countup

Countup resampling tries to select as many unique objects from each bin as possible before selecting the duplicates.

1. `num_global_objects_estimate_hist` objects are binned for each class using the configurations for resampling variable bins. This histogram is the initial estimate of the pdf of the objects of each class.
2. The new batch of objects is being read and after the cuts are applied `n_batch` objects remain. The objects are binned with the binning from step 1 and the reference number of the bin for each object is saved. Upscaling is not available for this method and setting `upscale_pdf` raises an error.
3. The number of **requested** objects in each bin are calculated as `floor(n_batch*flavour.sampling_fraction*pdf_target_flavour+uniform([0, 1]))` so that if `n_batch*flavour.sampling_fraction*pdf_target_flavour=1.2` it has a 80% chance to be rounded up to 1 and 20% chance to be rounded up to 2 so that for each bin we get an integer number that on average corresponds to the expected value. 
4. From each bin we select consecutively (without replacement) the required number of objects. If the bin holds less objects than the **requested** number, the rest of the objects in this bin is chosen at random from this bin with replacement. This way only few objects in each bin are repeated for `flavour.sampling_fraction=1` and rarely any are repeated for smaller sampling fractions
5. It may happen that we required objects from a bin that is empty in this batch, thus the operations above lead to less objects than **requested** in total. To compensate we resample this number at random (with replacement) from the already sampled objects. This leads to some additional repetitions but this way we can be sure that we adhere to the target pdf.


