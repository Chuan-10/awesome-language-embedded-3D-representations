# Awesome Language Embedded 3D Representations

A curated list of papers and open-source resources focused on language embedded 3D neural representations. If you have any suggestions or additions, please feel free to contribute.

The README template is borrowed from [MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting).

## Table of contents

- [Embedding Features](#embedding-features)
- [Embedding Labels](#embedding-labels)
- [Editing](#editing)



## Embedding Features

### [NIPS22] Decomposing NeRF for Editing via Feature Field Distillation

**Authors**: Sosuke Kobayashi, Eiichi Matsumoto, Vincent Sitzmann

**Resources:** [📄 Paper](https://arxiv.org/abs/2205.15585) | [🌐 Project Page](https://pfnet-research.github.io/distilled-feature-fields/) | [💻 Code](https://github.com/pfnet-research/distilled-feature-fields)

<details span>
<summary><b>Abstract</b></summary>
Emerging neural radiance fields (NeRF) are a promising scene representation for computer graphics, enabling high-quality 3D reconstruction and novel view synthesis from image observations. However, editing a scene represented by a NeRF is challenging, as the underlying connectionist representations such as MLPs or voxel grids are not object-centric or compositional. In particular, it has been difficult to selectively edit specific regions or objects. In this work, we tackle the problem of semantic scene decomposition of NeRFs to enable query-based local editing of the represented 3D scenes. We propose to distill the knowledge of off-the-shelf, self-supervised 2D image feature extractors such as CLIP-LSeg or DINO into a 3D feature field optimized in parallel to the radiance field. Given a user-specified query of various modalities such as text, an image patch, or a point-and-click selection, 3D feature fields semantically decompose 3D space without the need for re-training and enable us to semantically select and edit regions in the radiance field. Our experiments validate that the distilled feature fields (DFFs) can transfer recent progress in 2D vision and language foundation models to 3D scene representations, enabling convincing 3D segmentation and selective editing of emerging neural graphics representations.
</details>


###  [3DV22] Neural Feature Fusion Fields: 3D Distillation of Self-Supervised 2D Image Representations

**Authors**: Vadim Tschernezki, Iro Laina, Diane Larlus, Andrea Vedaldi

**Resources:**   [📄 Paper](https://arxiv.org/abs/2209.03494) | [🌐 Project Page](https://www.robots.ox.ac.uk/~vadim/n3f/) | [💻 Code](https://github.com/dichotomies/N3F)

<details span>
<summary><b>Abstract</b></summary>
We present Neural Feature Fusion Fields (N3F), a method that improves dense 2D image feature extractors when the latter are applied to the analysis of multiple images reconstructible as a 3D scene. Given an image feature extractor, for example pre-trained using self-supervision, N3F uses it as a teacher to learn a student network defined in 3D space. The 3D student network is similar to a neural radiance field that distills said features and can be trained with the usual differentiable rendering machinery. As a consequence, N3F is readily applicable to most neural rendering formulations, including vanilla NeRF and its extensions to complex dynamic scenes. We show that our method not only enables semantic understanding in the context of scene-specific neural fields without the use of manual labels, but also consistently improves over the self-supervised 2D baselines. This is demonstrated by considering various tasks, such as 2D object retrieval, 3D segmentation, and scene editing, in diverse sequences, including long egocentric videos in the EPIC-KITCHENS benchmark.
</details>


### [ICCV23] LERF: Language Embedded Radiance Fields

**Authors**: Justin Kerr\*, Chung Min Kim\*, Ken Goldberg, Angjoo Kanazawa, Matthew Tancik

**Resources:** [📄 Paper](https://arxiv.org/abs/2303.09553) | [🌐 Project Page](https://www.lerf.io/) | [💻 Code](https://github.com/kerrj/lerf) | [📦 Data](https://drive.google.com/drive/folders/1vh0mSl7v29yaGsxleadcj-LCZOE_WEWB?usp=sharing)

<details span>
<summary><b>Abstract</b></summary>
Humans describe the physical world using natural language to refer to specific 3D locations based on a vast range of properties: visual appearance, semantics, abstract associations, or actionable affordances. In this work we propose Language Embedded Radiance Fields (LERFs), a method for grounding language embeddings from off-the-shelf models like CLIP into NeRF, which enable these types of open-ended language queries in 3D. LERF learns a dense, multi-scale language field inside NeRF by volume rendering CLIP embeddings along training rays, supervising these embeddings across training views to provide multi-view consistency and smooth the underlying language field. After optimization, LERF can extract 3D relevancy maps for a broad range of language prompts interactively in real-time, which has potential use cases in robotics, understanding vision-language models, and interacting with 3D scenes. LERF enables pixel-aligned, zero-shot queries on the distilled 3D CLIP embeddings without relying on region proposals or masks, supporting long-tail open-vocabulary queries hierarchically across the volume.
</details>


### [ICCV23] FeatureNeRF: Learning Generalizable NeRFs by Distilling Foundation Models

**Authors**: Jianglong Ye, Naiyan Wang, Xiaolong Wang

**Resources:** [📄 Paper](https://arxiv.org/abs/2303.12786) | [🌐 Project Page](https://jianglongye.com/featurenerf/) | [💻 Code (not yet)]()

<details span>
<summary><b>Abstract</b></summary>
Recent works on generalizable NeRFs have shown promising results on novel view synthesis from single or few images. However, such models have rarely been applied on other downstream tasks beyond synthesis such as semantic understanding and parsing. In this paper, we propose a novel framework named FeatureNeRF to learn generalizable NeRFs by distilling pre-trained vision foundation models (e.g., DINO, Latent Diffusion). FeatureNeRF leverages 2D pre-trained foundation models to 3D space via neural rendering, and then extract deep features for 3D query points from NeRF MLPs. Consequently, it allows to map 2D images to continuous 3D semantic feature volumes, which can be used for various downstream tasks. We evaluate FeatureNeRF on tasks of 2D/3D semantic keypoint transfer and 2D/3D object part segmentation. Our extensive experiments demonstrate the effectiveness of FeatureNeRF as a generalizable 3D semantic feature extractor. 
</details>



### [NIPS23] 3D Open-vocabulary Segmentation with Foundation Models

**Authors**: Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, Shijian Lu

**Resources:** [📄 Paper](https://arxiv.org/abs/2305.14093) | [💻 Code](https://github.com/Kunhao-Liu/3D-OVS) | [📦 Data](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing)

<details span>
<summary><b>Abstract</b></summary>
Open-vocabulary segmentation of 3D scenes is a fundamental function of human perception and thus a crucial objective in computer vision research. However, this task is heavily impeded by the lack of large-scale and diverse 3D open-vocabulary segmentation datasets for training robust and generalizable models. Distilling knowledge from pre-trained 2D open-vocabulary segmentation models helps but it compromises the open-vocabulary feature as the 2D models are mostly finetuned with close-vocabulary datasets. We tackle the challenges in 3D open-vocabulary segmentation by exploiting pre-trained foundation models CLIP and DINO in a weakly supervised manner. Specifically, given only the open-vocabulary text descriptions of the objects in a scene, we distill the open-vocabulary multimodal knowledge and object reasoning capability of CLIP and DINO into a neural radiance field (NeRF), which effectively lifts 2D features into view-consistent 3D segmentation. A notable aspect of our approach is that it does not require any manual segmentation annotations for either the foundation models or the distillation process. Extensive experiments show that our method even outperforms fully supervised models trained with segmentation annotations in certain scenes, suggesting that 3D open-vocabulary segmentation can be effectively learned from 2D images and text-image pairs.
</details>


### [arXiv2312] Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding 
**Authors**: Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, Shao-Hua Guan

**Resources:** [📄 Paper](https://arxiv.org/pdf/2311.18482.pdf) | [🌐 Project Page](https://buaavrcg.github.io/LEGaussians/) | [💻 Code (not yet)]()

<details span>
<summary><b>Abstract</b></summary>
Open-vocabulary querying in 3D space is challenging but essential for scene understanding tasks such as object localization and segmentation. Language-embedded scene representations have made progress by incorporating language features into 3D spaces. However, their efficacy heavily depends on neural networks that are resource-intensive in training and rendering. Although recent 3D Gaussians offer efficient and high-quality novel view synthesis, directly embedding language features in them leads to prohibitive memory usage and decreased performance. In this work, we introduce Language Embedded 3D Gaussians, a novel scene representation for open-vocabulary query tasks. Instead of embedding high-dimensional raw semantic features on 3D Gaussians, we propose a dedicated quantization scheme that drastically alleviates the memory requirement, and a novel embedding procedure that achieves smoother yet high accuracy query, countering the multi-view feature inconsistencies and the high-frequency inductive bias in point-based representations. Our comprehensive experiments show that our representation achieves the best visual quality and language querying accuracy across current language-embedded representations, while maintaining real-time rendering frame rates on a single desktop GPU. 
</details>



## Embedding Labels

### [ICCV21] In-Place Scene Labelling and Understanding with Implicit Scene Representation

**Authors**: Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, Andrew Davison

**Resources:**   [📄 Paper](https://arxiv.org/abs/2103.15875) | [🌐 Project Page](https://shuaifengzhi.com/Semantic-NeRF/) | [💻 Code](https://github.com/Harry-Zhi/semantic_nerf/) | [📦 Data](https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0) | [🎬 Video](https://youtu.be/FpShWO7LVbM)

<details span>
<summary><b>Abstract</b></summary>
Semantic labelling is highly correlated with geometry and radiance reconstruction, as scene entities with similar shape and appearance are more likely to come from similar classes. Recent implicit neural reconstruction techniques are appealing as they do not require prior training data, but the same fully self-supervised approach is not possible for semantics because labels are human-defined properties. We extend neural radiance fields (NeRF) to jointly encode semantics with appearance and geometry, so that complete and accurate 2D semantic labels can be achieved using a small amount of in-place annotations specific to the scene. The intrinsic multi-view consistency and smoothness of NeRF benefit semantics by enabling sparse labels to efficiently propagate. We show the benefit of this approach when labels are either sparse or very noisy in room-scale scenes. We demonstrate its advantageous properties in various interesting applications such as an efficient scene labelling tool, novel semantic view synthesis, label denoising, super-resolution, label interpolation and multi-view semantic label fusion in visual semantic mapping systems.
</details>


### [CVPR23] Panoptic Lifting for 3D Scene Understanding with Neural Fields

**Authors**: Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Buló, Norman Müller, Matthias Nießner, Angela Dai, Peter Kontschieder

**Resources:**   [📄 Paper](https://arxiv.org/abs/2212.09802) | [🌐 Project Page](https://nihalsid.github.io/panoptic-lifting/) | [💻 Code](https://github.com/nihalsid/panoptic-lifting) | [🎬 Video](https://youtu.be/QtsiL-6rSuM)

<details span>
<summary><b>Abstract</b></summary>
We propose Panoptic Lifting, a novel approach for learning panoptic 3D volumetric representations from images of in-the-wild scenes. Once trained, our model can render color images together with 3D-consistent panoptic segmentation from novel viewpoints. Unlike existing approaches which use 3D input directly or indirectly, our method requires only machine-generated 2D panoptic segmentation masks inferred from a pre-trained network. Our core contribution is a panoptic lifting scheme based on a neural field representation that generates a unified and multi-view consistent, 3D panoptic representation of the scene. To account for inconsistencies of 2D instance identifiers across views, we solve a linear assignment with a cost based on the model's current predictions and the machine-generated segmentation masks, thus enabling us to lift 2D instances to 3D in a consistent way. We further propose and ablate contributions that make our method more robust to noisy, machine-generated labels, including test-time augmentations for confidence estimates, segment consistency loss, bounded segmentation fields, and gradient stopping. Experimental results validate our approach on the challenging Hypersim, Replica, and ScanNet datasets, improving by 8.4, 13.8, and 10.6% in scene-level PQ over state of the art.
</details>


## Editing








[CVPR23] Nerflets: Local Radiance Fields for Efficient Structure-Aware 3D Scene Representation from 2D Supervision

