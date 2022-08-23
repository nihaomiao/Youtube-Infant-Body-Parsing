# Youtube-Infant-Body-Parsing
The infant video dataset collected from Youtube with body parsing annotations.

Dataset Description
-----
This dataset includes 90 infant movement videos collected from Youtube by [Chambers et al. (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8011647/). URLs of videos are provided in their [github](https://github.com/cchamber/Infant_movement_assessment). We downloaded videos and downsampled them every 4 frames due to the long length of original videos. Sampled frames are available on this [google drive](https://drive.google.com/file/d/1sm5Ril_2YT3cidSkCpL6EcXZPFf2eNSW/view?usp=sharing). The format of each frame name is `1{6_digits_of_video_index}{6_digits_of_frame_index}`. For infant body parsing, we collect annotations for five classes: background, head, arm, torso and leg. We randomly split all videos into 68 training videos and 22 testing videos, resulting in 2,149 labeled and 4,690 unlabeled training frames, and 1,256 labeled and 2,737 unlabeled testing frames. They corresponds to `train_label.json`, `train_unlabel.json`, `test_label.json`, and `test_unlabel.json`, respectively. The following codes show how to use these json files.
