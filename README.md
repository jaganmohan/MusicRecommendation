

# MusicRecommendation

Final project for course **Big Data Ecosystems**, University of Florida.

## Abstract

This project discuss the task of recommending songs by predicticting latent factors from their mel-spectrograms.

## Introduction


## The Dataset

## Preprocessing the data

 
## The Model
 
 
## Results


### Documentation

• previewDownloader.py: 
USAGE: python previewDownloader.py [path to MSD data] 
This script iterate over all ‘.h5’ in a directory and download a 30 seconds sample from 7digital.

• preproccess.py: 
USAGE: python preproccess.py [path to MSD mp3 data] 
This script pre-processing the sound files. Calculating MFCC for a sliding window and saving the result in a ‘.pp’ file.

• formatInput.py: 
USAGE: python formatInput.py [path to MSD pp data] 
The script iterates over all ‘.pp’ files and generates ‘data’ and ‘labels’ that will be used as an input to the NN. 
Moreover, the script output a t-SNE graph at the end.

• train.py: 
USAGE: python train.py 
This script builds the neural network and feeds it with ‘data’ and ‘labels’.  When it is done it will save ‘model.final’.

### Complete Installation

<ul>
<li>Download the dataset files from https://www.dropbox.com/s/8ohx6m23co1qaz3/DataSet.zip?dl=0.</li>
<li>Unzip file</li>
<li>Place dataset files in the structure they are ordered in</li>
</ul>


## References

[1] Tao Feng, Deep learning for music genre classification, University of Illinois. https://courses.engr.illinois.edu/ece544na/fa2014/Tao_Feng.pdf
[2]Aar̈onvandenOord,SanderDieleman,BenjaminSchrauwen,Deepcontent- based music recommendation. http://papers.nips.cc/paper/5004-deep-content-based- music-recommendation.pdf
[3] SANDER DIELEMAN, RECOMMENDING MUSIC ON SPOTIFY WITH DEEP LEARNING, AUGUST 05, 2014. http://benanne.github.io/2014/08/05/spotify-cnns.html
[4] https://www.tensorflow.org
[5] GTZAN Genre Collection. http://marsyasweb.appspot.com/download/ data_sets/
[6] Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011. http://
labrosa.ee.columbia.edu/millionsong/
[7] Hendrik Schreiber. Improving genre annotations for the million song dataset. In
Proceedings of the 16th International Conference on Music Information Retrieval (IS- MIR), pages 241-247, 2015.
http://www.tagtraum.com/msd_genre_datasets.html
[8] https://www.7digital.com
[9] https://github.com/bmcfee/librosa
[10] http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html [11] Helge Homburg, Ingo Mierswa, Bu l̈ent Mo l̈ler, Katharina Morik and Michael
Wurst, A BENCHMARK DATASET FOR AUDIO CLASSIFICATION AND CLUSTERING, University of Dortmund, AI Unit. http://sfb876.tu-dortmund.de/PublicPublicationFiles/ homburg_etal_2005a.pdf

## Contributors

Kyla Gardner, Subash Nerella, Jaganmohan M
