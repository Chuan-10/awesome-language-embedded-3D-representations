# Awesome Language Embedded 3D Representations [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of papers and open-source resources focused on language embedded 3D neural representations. This project is under construction. If you have any suggestions or additions, please feel free to contribute. 

The README template is borrowed from [MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting).


## Table of contents

- [For Scene Understanding](#for-scene-understanding)
- [For Segmentation](#for-segmentation)
- [For Editing](#for-editing)
- [SLAM](#slam)
- [Resources](#resources)

<details span>
<summary><b>Update Log:</b></summary>
<br>

**August 3, 2024**: 
   * 1 paper added: NIS-SLAM

**August 2, 2024**: 
   * 1 paper added: Semantic Gaussians

**July 18, 2024**: 
   * 1 paper added: Click-Gaussian

**July 15, 2024**: 
   * 1 paper added: SA4D

**July 9, 2024**: 
   * 1 paper added: MaskField

**June 24, 2024**: 
   * 1 paper added: Gaga

**June 6, 2024**: 
   * 1 paper added: OpenGaussian

**May 27, 2024**: 
   * 1 paper added: TIGER

**April 8, 2024**: 
   * 1 paper added: OpenNeRF

**April 7, 2024**: 
   * 3 papers added: ConceptFusion, GSNeRF, SNI-SLAM
   * Codes released: LEGausssians
   * Data released: LEGausssians, GARField


**March 12, 2024**: 
   * 2 papers removed: GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions, GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting 
   * 4 papers' status updated.
   * Structure modified.

**January 20, 2024**: 
   * Codes released: Gaussian Grouping, Feature 3DGS and Segment Any 3D Gaussians.
   * 2 papers added: FMGS and GARField.
   * Update Log added.

</details>

<br>


## For Scene Understanding

### [ICCV21] In-Place Scene Labelling and Understanding with Implicit Scene Representation

**Authors**: Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, Andrew Davison

**Resources:**   [📄 Paper](https://arxiv.org/pdf/2103.15875.pdf) | [🌐 Project Page](https://shuaifengzhi.com/Semantic-NeRF/) | [💻 Code](https://github.com/Harry-Zhi/semantic_nerf/) | [📦 Data](https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0) | [🎥 Video](https://youtu.be/FpShWO7LVbM)

<details span>
<summary><b>Abstract</b></summary>
Semantic labelling is highly correlated with geometry and radiance reconstruction, as scene entities with similar shape and appearance are more likely to come from similar classes. Recent implicit neural reconstruction techniques are appealing as they do not require prior training data, but the same fully self-supervised approach is not possible for semantics because labels are human-defined properties. We extend neural radiance fields (NeRF) to jointly encode semantics with appearance and geometry, so that complete and accurate 2D semantic labels can be achieved using a small amount of in-place annotations specific to the scene. The intrinsic multi-view consistency and smoothness of NeRF benefit semantics by enabling sparse labels to efficiently propagate. We show the benefit of this approach when labels are either sparse or very noisy in room-scale scenes. We demonstrate its advantageous properties in various interesting applications such as an efficient scene labelling tool, novel semantic view synthesis, label denoising, super-resolution, label interpolation and multi-view semantic label fusion in visual semantic mapping systems.
</details>


###  [3DV22] Neural Feature Fusion Fields: 3D Distillation of Self-Supervised 2D Image Representations

**Authors**: Vadim Tschernezki, Iro Laina, Diane Larlus, Andrea Vedaldi

**Resources:**   [📄 Paper](https://arxiv.org/pdf/2209.03494.pdf) | [🌐 Project Page](https://www.robots.ox.ac.uk/~vadim/n3f/) | [💻 Code](https://github.com/dichotomies/N3F)

<details span>
<summary><b>Abstract</b></summary>
We present Neural Feature Fusion Fields (N3F), a method that improves dense 2D image feature extractors when the latter are applied to the analysis of multiple images reconstructible as a 3D scene. Given an image feature extractor, for example pre-trained using self-supervision, N3F uses it as a teacher to learn a student network defined in 3D space. The 3D student network is similar to a neural radiance field that distills said features and can be trained with the usual differentiable rendering machinery. As a consequence, N3F is readily applicable to most neural rendering formulations, including vanilla NeRF and its extensions to complex dynamic scenes. We show that our method not only enables semantic understanding in the context of scene-specific neural fields without the use of manual labels, but also consistently improves over the self-supervised 2D baselines. This is demonstrated by considering various tasks, such as 2D object retrieval, 3D segmentation, and scene editing, in diverse sequences, including long egocentric videos in the EPIC-KITCHENS benchmark.
</details>


### [CVPR23] Panoptic Lifting for 3D Scene Understanding with Neural Fields

**Authors**: Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Buló, Norman Müller, Matthias Nießner, Angela Dai, Peter Kontschieder

**Resources:**   [📄 Paper](https://arxiv.org/pdf/2212.09802.pdf) | [🌐 Project Page](https://nihalsid.github.io/panoptic-lifting/) | [💻 Code](https://github.com/nihalsid/panoptic-lifting) | [🎥 Video](https://youtu.be/QtsiL-6rSuM)

<details span>
<summary><b>Abstract</b></summary>
We propose Panoptic Lifting, a novel approach for learning panoptic 3D volumetric representations from images of in-the-wild scenes. Once trained, our model can render color images together with 3D-consistent panoptic segmentation from novel viewpoints. Unlike existing approaches which use 3D input directly or indirectly, our method requires only machine-generated 2D panoptic segmentation masks inferred from a pre-trained network. Our core contribution is a panoptic lifting scheme based on a neural field representation that generates a unified and multi-view consistent, 3D panoptic representation of the scene. To account for inconsistencies of 2D instance identifiers across views, we solve a linear assignment with a cost based on the model's current predictions and the machine-generated segmentation masks, thus enabling us to lift 2D instances to 3D in a consistent way. We further propose and ablate contributions that make our method more robust to noisy, machine-generated labels, including test-time augmentations for confidence estimates, segment consistency loss, bounded segmentation fields, and gradient stopping. Experimental results validate our approach on the challenging Hypersim, Replica, and ScanNet datasets, improving by 8.4, 13.8, and 10.6% in scene-level PQ over state of the art.
</details>


### [CVPR23] Nerflets: Local Radiance Fields for Efficient Structure-Aware 3D Scene Representation from 2D Supervision

**Authors**: Xiaoshuai Zhang, Abhijit Kundu, Thomas Funkhouser, Leonidas Guibas, Hao Su, Kyle Genova

**Resources:**   [📄 Paper](https://arxiv.org/pdf/2303.03361.pdf)

<details span>
<summary><b>Abstract</b></summary>
We address efficient and structure-aware 3D scene representation from images. Nerflets are our key contribution -- a set of local neural radiance fields that together represent a scene. Each nerflet maintains its own spatial position, orientation, and extent, within which it contributes to panoptic, density, and radiance reconstructions. By leveraging only photometric and inferred panoptic image supervision, we can directly and jointly optimize the parameters of a set of nerflets so as to form a decomposed representation of the scene, where each object instance is represented by a group of nerflets. During experiments with indoor and outdoor environments, we find that nerflets: (1) fit and approximate the scene more efficiently than traditional global NeRFs, (2) allow the extraction of panoptic and photometric renderings from arbitrary views, and (3) enable tasks rare for NeRFs, such as 3D panoptic segmentation and interactive editing.
</details>


### [ICCV23] LERF: Language Embedded Radiance Fields

**Authors**: Justin Kerr\*, Chung Min Kim\*, Ken Goldberg, Angjoo Kanazawa, Matthew Tancik

**Resources:** [📄 Paper](https://arxiv.org/pdf/2303.09553.pdf) | [🌐 Project Page](https://www.lerf.io/) | [💻 Code](https://github.com/kerrj/lerf) | [📦 Data](https://drive.google.com/drive/folders/1vh0mSl7v29yaGsxleadcj-LCZOE_WEWB?usp=sharing)

<details span>
<summary><b>Abstract</b></summary>
Humans describe the physical world using natural language to refer to specific 3D locations based on a vast range of properties: visual appearance, semantics, abstract associations, or actionable affordances. In this work we propose Language Embedded Radiance Fields (LERFs), a method for grounding language embeddings from off-the-shelf models like CLIP into NeRF, which enable these types of open-ended language queries in 3D. LERF learns a dense, multi-scale language field inside NeRF by volume rendering CLIP embeddings along training rays, supervising these embeddings across training views to provide multi-view consistency and smooth the underlying language field. After optimization, LERF can extract 3D relevancy maps for a broad range of language prompts interactively in real-time, which has potential use cases in robotics, understanding vision-language models, and interacting with 3D scenes. LERF enables pixel-aligned, zero-shot queries on the distilled 3D CLIP embeddings without relying on region proposals or masks, supporting long-tail open-vocabulary queries hierarchically across the volume.
</details>


### [RSS23] ConceptFusion: Open-set Multimodal 3D Mapping

**Authors**: Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam Paull, Florian Shkurti, Antonio Torralba

**Resources:**   [📄 Paper](https://concept-fusion.github.io/assets/pdf/2023-ConceptFusion.pdf) | [🌐 Project Page](https://concept-fusion.github.io/) | [💻 Code](https://github.com/concept-fusion/concept-fusion) | [📦 Data (not yet)]() | [🎥 Short Presentation](https://www.youtube.com/watch?v=rkXgws8fiDs)

<details span>
<summary><b>Abstract</b></summary>
Building 3D maps of the environment is central to robot navigation, planning, and interaction with objects in a scene. Most existing approaches that integrate semantic concepts with 3D maps largely remain confined to the closed-set setting: they can only reason about a finite set of concepts, pre-defined at training time. Further, these maps can only be queried using class labels, or in recent work, using text prompts.
We address both these issues with ConceptFusion, a scene representation that is (1) fundamentally open-set, enabling reasoning beyond a closed set of concepts and (ii) inherently multimodal, enabling a diverse range of possible queries to the 3D map, from language, to images, to audio, to 3D geometry, all working in concert. ConceptFusion leverages the open-set capabilities of today's foundation models pre-trained on internet-scale data to reason about concepts across modalities such as natural language, images, and audio. We demonstrate that pixel-aligned open-set features can be fused into 3D maps via traditional SLAM and multi-view fusion approaches. This enables effective zero-shot spatial reasoning, not needing any additional training or finetuning, and retains long-tailed concepts better than supervised approaches, outperforming them by more than 40% margin on 3D IoU. We extensively evaluate ConceptFusion on a number of real-world datasets, simulated home environments, a real-world tabletop manipulation task, and an autonomous driving platform. We showcase new avenues for blending foundation models with 3D open-set multimodal mapping.
</details>


### [ICCV23] FeatureNeRF: Learning Generalizable NeRFs by Distilling Foundation Models

**Authors**: Jianglong Ye, Naiyan Wang, Xiaolong Wang

**Resources:** [📄 Paper](https://arxiv.org/pdf/2303.12786.pdf) | [🌐 Project Page](https://jianglongye.com/featurenerf/) | [💻 Code (not yet)]()

<details span>
<summary><b>Abstract</b></summary>
Recent works on generalizable NeRFs have shown promising results on novel view synthesis from single or few images. However, such models have rarely been applied on other downstream tasks beyond synthesis such as semantic understanding and parsing. In this paper, we propose a novel framework named FeatureNeRF to learn generalizable NeRFs by distilling pre-trained vision foundation models (e.g., DINO, Latent Diffusion). FeatureNeRF leverages 2D pre-trained foundation models to 3D space via neural rendering, and then extract deep features for 3D query points from NeRF MLPs. Consequently, it allows to map 2D images to continuous 3D semantic feature volumes, which can be used for various downstream tasks. We evaluate FeatureNeRF on tasks of 2D/3D semantic keypoint transfer and 2D/3D object part segmentation. Our extensive experiments demonstrate the effectiveness of FeatureNeRF as a generalizable 3D semantic feature extractor. 
</details>


### [CVPR24] Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields

**Authors**: Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, Achuta Kadambi 

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2312.03203.pdf) | [🌐 Project Page](https://feature-3dgs.github.io/) | [💻 Code](https://github.com/ShijieZhou-UCLA/feature-3dgs) | [🎥 Short Presentation](https://www.youtube.com/watch?v=YWZiF-WvMN4&t=4s)

<details span>
<summary><b>Abstract</b></summary>
3D scene representations have gained immense popularity in recent years. Methods that use Neural Radiance fields are versatile for traditional tasks such as novel view synthesis. In recent times, some work has emerged that aims to extend the functionality of NeRF beyond view synthesis, for semantically aware tasks such as editing and segmentation using 3D feature field distillation from 2D foundation models. However, these methods have two major limitations: (a) they are limited by the rendering speed of NeRF pipelines, and (b) implicitly represented feature fields suffer from continuity artifacts reducing feature quality. Recently, 3D Gaussian Splatting has shown state-of-the-art performance on real-time radiance field rendering. In this work, we go one step further: in addition to radiance field rendering, we enable 3D Gaussian splatting on arbitrary-dimension semantic features via 2D foundation model distillation. This translation is not straightforward: naively incorporating feature fields in the 3DGS framework leads to warp-level divergence. We propose architectural and training changes to efficiently avert this problem. Our proposed method is general, and our experiments showcase novel view semantic segmentation, language-guided editing and segment anything through learning feature fields from state-of-the-art 2D foundation models such as SAM and CLIP-LSeg. Across experiments, our distillation method is able to provide comparable or better results, while being significantly faster to both train and render. Additionally, to the best of our knowledge, we are the first method to enable point and bounding-box prompting for radiance field manipulation, by leveraging the SAM model. 
</details>


### [CVPR24] LEGaussians: Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding 

**Authors**: Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, Shao-Hua Guan

**Resources:** [📄 Paper](https://arxiv.org/pdf/2311.18482.pdf) | [🌐 Project Page](https://buaavrcg.github.io/LEGaussians/) | [💻 Code](https://github.com/buaavrcg/LEGaussians)  | [📦 Data](https://drive.google.com/drive/folders/1vJ3le9lIGq8zl3ls1OzkBQ-rXLiSSc22)


<details span>
<summary><b>Abstract</b></summary>
Open-vocabulary querying in 3D space is challenging but essential for scene understanding tasks such as object localization and segmentation. Language-embedded scene representations have made progress by incorporating language features into 3D spaces. However, their efficacy heavily depends on neural networks that are resource-intensive in training and rendering. Although recent 3D Gaussians offer efficient and high-quality novel view synthesis, directly embedding language features in them leads to prohibitive memory usage and decreased performance. In this work, we introduce Language Embedded 3D Gaussians, a novel scene representation for open-vocabulary query tasks. Instead of embedding high-dimensional raw semantic features on 3D Gaussians, we propose a dedicated quantization scheme that drastically alleviates the memory requirement, and a novel embedding procedure that achieves smoother yet high accuracy query, countering the multi-view feature inconsistencies and the high-frequency inductive bias in point-based representations. Our comprehensive experiments show that our representation achieves the best visual quality and language querying accuracy across current language-embedded representations, while maintaining real-time rendering frame rates on a single desktop GPU. 
</details>


### [CVPR24] LangSplat: 3D Language Gaussian Splatting 

**Authors**: Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, Hanspeter Pfister 

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2312.16084.pdf) | [🌐 Project Page](https://langsplat.github.io/) | [💻 Code](https://github.com/minghanqin/LangSplat) | [🎥 Short Presentation](https://www.youtube.com/watch?v=XMlyjsei-Es)

<details span>
<summary><b>Abstract</b></summary>
Human lives in a 3D world and commonly uses natural language to interact with a 3D scene. Modeling a 3D language field to support open-ended language queries in 3D has gained increasing attention recently. This paper introduces LangSplat, which constructs a 3D language field that enables precise and efficient open-vocabulary querying within 3D spaces. Unlike existing methods that ground CLIP language embeddings in a NeRF model, LangSplat advances the field by utilizing a collection of 3D Gaussians, each encoding language features distilled from CLIP, to represent the language field. By employing a tile-based splatting technique for rendering language features, we circumvent the costly rendering process inherent in NeRF. Instead of directly learning CLIP embeddings, LangSplat first trains a scene-wise language autoencoder and then learns language features on the scene-specific latent space, thereby alleviating substantial memory demands imposed by explicit modeling. Existing methods struggle with imprecise and vague 3D language fields, which fail to discern clear boundaries between objects. We delve into this issue and propose to learn hierarchical semantics using SAM, thereby eliminating the need for extensively querying the language field across various scales and the regularization of DINO features. Extensive experiments on open-vocabulary 3D object localization and semantic segmentation demonstrate that LangSplat significantly outperforms the previous state-of-the-art method LERF by a large margin. Notably, LangSplat is extremely efficient, achieving a {\speed} × speedup compared to LERF at the resolution of 1440 × 1080.
</details>


### [IJCV24] FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding 

**Authors**: Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, Mingyang Li

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2401.01970.pdf) | [🌐 Project Page](https://xingxingzuo.github.io/fmgs/)

<details span>
<summary><b>Abstract</b></summary>
Precisely perceiving the geometric and semantic properties of real-world 3D objects is crucial for the continued evolution of augmented reality and robotic applications. To this end, we present \algfull{} (\algname{}), which incorporates vision-language embeddings of foundation models into 3D Gaussian Splatting (GS). The key contribution of this work is an efficient method to reconstruct and represent 3D vision-language models. This is achieved by distilling feature maps generated from image-based foundation models into those rendered from our 3D model. To ensure high-quality rendering and fast training, we introduce a novel scene representation by integrating strengths from both GS and multi-resolution hash encodings (MHE). Our effective training procedure also introduces a pixel alignment loss that makes the rendered feature distance of same semantic entities close, following the pixel-level semantic boundaries. Our results demonstrate remarkable multi-view semantic consistency, facilitating diverse downstream tasks, beating state-of-the-art methods by 10.2 percent on open-vocabulary language-based object detection, despite that we are 851× faster for inference. This research explores the intersection of vision, language, and 3D scene representation, paving the way for enhanced scene understanding in uncontrolled real-world environments.
</details>


### [CVPR24] GARField: Group Anything with Radiance Fields 

**Authors**: Chung Min Kim, Mingxuan Wu, Justin Kerr, Ken Goldberg, Matthew Tancik, Angjoo Kanazawa

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2401.09419.pdf) | [🌐 Project Page](https://www.garfield.studio/) | [💻 Code](https://github.com/chungmin99/garfield) | [📦 Data](https://drive.google.com/drive/folders/1LDvbFTQuaQxru5ELsfCjX7sTkg1WotX0)

<details span>
<summary><b>Abstract</b></summary>
Grouping is inherently ambiguous due to the multiple levels of granularity in which one can decompose a scene -- should the wheels of an excavator be considered separate or part of the whole? We present Group Anything with Radiance Fields (GARField), an approach for decomposing 3D scenes into a hierarchy of semantically meaningful groups from posed image inputs. To do this we embrace group ambiguity through physical scale: by optimizing a scale-conditioned 3D affinity feature field, a point in the world can belong to different groups of different sizes. We optimize this field from a set of 2D masks provided by Segment Anything (SAM) in a way that respects coarse-to-fine hierarchy, using scale to consistently fuse conflicting masks from different viewpoints. From this field we can derive a hierarchy of possible groupings via automatic tree construction or user interaction. We evaluate GARField on a variety of in-the-wild scenes and find it effectively extracts groups at many levels: clusters of objects, objects, and various subparts. GARField inherently represents multi-view consistent groupings and produces higher fidelity groups than the input SAM masks. GARField's hierarchical grouping could have exciting downstream applications such as 3D asset extraction or dynamic scene understanding. See the project website at https://www.garfield.studio/
</details>


### [CVPR24] GSNeRF: Generalizable Semantic Neural Radiance Fields with Enhanced 3D Scene Understanding

**Authors**: Zi-Ting Chou, Sheng-Yu Huang, I-Jieh Liu, Yu-Chiang Frank Wang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2403.03608.pdf)

<details span>
<summary><b>Abstract</b></summary>
Utilizing multi-view inputs to synthesize novel-view images, Neural Radiance Fields (NeRF) have emerged as a popular research topic in 3D vision. In this work, we introduce a Generalizable Semantic Neural Radiance Field (GSNeRF), which uniquely takes image semantics into the synthesis process so that both novel view images and the associated semantic maps can be produced for unseen scenes. Our GSNeRF is composed of two stages: Semantic Geo-Reasoning and Depth-Guided Visual rendering. The former is able to observe multi-view image inputs to extract semantic and geometry features from a scene. Guided by the resulting image geometry information, the latter performs both image and semantic rendering with improved performances. Our experiments not only confirm that GSNeRF performs favorably against prior works on both novel-view image and semantic segmentation synthesis but the effectiveness of our sampling strategy for visual rendering is further verified.
</details>


### [arXiv2403] Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting

**Authors**: Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, Qing Li

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2403.15624) | [🌐 Project Page](https://semantic-gaussians.github.io) | [💻 Code](https://github.com/sharinka0715/semantic-gaussians)

<details span>
<summary><b>Abstract</b></summary>
Open-vocabulary 3D scene understanding presents a significant challenge in computer vision, withwide-ranging applications in embodied agents and augmented reality systems. Previous approaches haveadopted Neural Radiance Fields (NeRFs) to analyze 3D scenes. In this paper, we introduce SemanticGaussians, a novel open-vocabulary scene understanding approach based on 3D Gaussian Splatting. Our keyidea is distilling pre-trained 2D semantics into 3D Gaussians. We design a versatile projection approachthat maps various 2Dsemantic features from pre-trained image encoders into a novel semantic component of 3D Gaussians, withoutthe additional training required by NeRFs. We further build a 3D semantic network that directly predictsthe semantic component from raw 3D Gaussians for fast inference. We explore several applications ofSemantic Gaussians: semantic segmentation on ScanNet-20, where our approach attains a 4.2% mIoU and 4.0%mAcc improvement over prior open-vocabulary scene understanding counterparts; object part segmentation,sceneediting, and spatial-temporal segmentation with better qualitative results over 2D and 3D baselines,highlighting its versatility and effectiveness on supporting diverse downstream tasks.
</details>


### [arXiv2406] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding

**Authors**: Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, Jian Zhang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2406.02058) | [🌐 Project Page](https://3d-aigc.github.io/OpenGaussian/) | [💻 Code (not yet)]()

<details span>
<summary><b>Abstract</b></summary>
This paper introduces OpenGaussian, a method based on 3D Gaussian Splatting (3DGS) capable of 3D point-level open vocabulary understanding. Our primary motivation stems from observing that existing 3DGS-based open vocabulary methods mainly focus on 2D pixel-level parsing. These methods struggle with 3D point-level tasks due to weak feature expressiveness and inaccurate 2D-3D feature associations. To ensure robust feature presentation and 3D point-level understanding, we first employ SAM masks without cross-frame associations to train instance features with 3D consistency. These features exhibit both intra-object consistency and inter-object distinction. Then, we propose a two-stage codebook to discretize these features from coarse to fine levels. At the coarse level, we consider the positional information of 3D points to achieve location-based clustering, which is then refined at the fine level. Finally, we introduce an instance-level 3D-2D feature association method that links 3D points to 2D masks, which are further associated with 2D CLIP features. Extensive experiments, including open vocabulary-based 3D object selection, 3D point cloud understanding, click-based 3D object selection, and ablation studies, demonstrate the effectiveness of our proposed method.
</details>

<br>


## For Segmentation

### [NIPS23] 3D Open-vocabulary Segmentation with Foundation Models

**Authors**: Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, Shijian Lu

**Resources:** [📄 Paper](https://arxiv.org/pdf/2305.14093.pdf) | [💻 Code](https://github.com/Kunhao-Liu/3D-OVS) | [📦 Data](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing)

<details span>
<summary><b>Abstract</b></summary>
Open-vocabulary segmentation of 3D scenes is a fundamental function of human perception and thus a crucial objective in computer vision research. However, this task is heavily impeded by the lack of large-scale and diverse 3D open-vocabulary segmentation datasets for training robust and generalizable models. Distilling knowledge from pre-trained 2D open-vocabulary segmentation models helps but it compromises the open-vocabulary feature as the 2D models are mostly finetuned with close-vocabulary datasets. We tackle the challenges in 3D open-vocabulary segmentation by exploiting pre-trained foundation models CLIP and DINO in a weakly supervised manner. Specifically, given only the open-vocabulary text descriptions of the objects in a scene, we distill the open-vocabulary multimodal knowledge and object reasoning capability of CLIP and DINO into a neural radiance field (NeRF), which effectively lifts 2D features into view-consistent 3D segmentation. A notable aspect of our approach is that it does not require any manual segmentation annotations for either the foundation models or the distillation process. Extensive experiments show that our method even outperforms fully supervised models trained with segmentation annotations in certain scenes, suggesting that 3D open-vocabulary segmentation can be effectively learned from 2D images and text-image pairs.
</details>


### [NIPS23] Segment Anything in 3D with NeRFs

**Authors**: Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Chen Yang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2304.12308.pdf) | [🌐 Project Page](https://jumpat.github.io/SA3D/) | [💻 Code](https://github.com/Jumpat/SegmentAnythingin3D?tab=readme-ov-file)

<details span>
<summary><b>Abstract</b></summary>
Interactive 3D segmentation in radiance fields is an appealing task since its importance in 3D scene understanding and manipulation. However, existing methods face challenges in either achieving fine-grained, multi-granularity segmentation or contending with substantial computational overhead, inhibiting real-time interaction. In this paper, we introduce Segment Any 3D GAussians (SAGA), a novel 3D interactive segmentation approach that seamlessly blends a 2D segmentation foundation model with 3D Gaussian Splatting (3DGS), a recent breakthrough of radiance fields. SAGA efficiently embeds multi-granularity 2D segmentation results generated by the segmentation foundation model into 3D Gaussian point features through well-designed contrastive training. Evaluation on existing benchmarks demonstrates that SAGA can achieve competitive performance with state-of-the-art methods. Moreover, SAGA achieves multi-granularity segmentation and accommodates various prompts, including points, scribbles, and 2D masks. Notably, SAGA can finish the 3D segmentation within milliseconds, achieving nearly 1000× acceleration1 compared to previous SOTA.
</details>


### [arXiv2312] Segment Any 3D Gaussians

**Authors**: Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, Qi Tian

**Resources:**  [📄 Paper](https://jumpat.github.io/SAGA/SAGA_paper.pdf) | [🌐 Project Page](https://jumpat.github.io/SAGA/) | [💻 Code](https://github.com/Jumpat/SegAnyGAussians)

<details span>
<summary><b>Abstract</b></summary>
Interactive 3D segmentation in radiance fields is an appealing task since its importance in 3D scene understanding and manipulation. However, existing methods face challenges in either achieving fine-grained, multi-granularity segmentation or contending with substantial computational overhead, inhibiting real-time interaction. In this paper, we introduce Segment Any 3D GAussians (SAGA), a novel 3D interactive segmentation approach that seamlessly blends a 2D segmentation foundation model with 3D Gaussian Splatting (3DGS), a recent breakthrough of radiance fields. SAGA efficiently embeds multi-granularity 2D segmentation results generated by the segmentation foundation model into 3D Gaussian point features through well-designed contrastive training. Evaluation on existing benchmarks demonstrates that SAGA can achieve competitive performance with state-of-the-art methods. Moreover, SAGA achieves multi-granularity segmentation and accommodates various prompts, including points, scribbles, and 2D masks. Notably, SAGA can finish the 3D segmentation within milliseconds, achieving nearly 1000× acceleration1 compared to previous SOTA.
</details>


### [ICLR24] OpenNeRF: OpenSet 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views

**Authors**: Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, Federico Tombari

**Resources:**  [📄 Paper](https://openreview.net/pdf?id=SgjAojPKb3) | [🌐 Project Page](https://opennerf.github.io/) | [💻 Code](https://github.com/opennerf/opennerf)

<details span>
<summary><b>Abstract</b></summary>
Large visual-language models (VLMs), like CLIP, enable open-set image segmentation to segment arbitrary concepts from an image in a zero-shot manner. This goes beyond the traditional closed-set assumption, i.e., where models can only segment classes from a pre-defined training set. More recently, first works on open-set segmentation in 3D scenes have appeared in the literature. These methods are heavily influenced by closed-set 3D convolutional approaches that process point clouds or polygon meshes. However, these 3D scene representations do not align well with the image-based nature of the visual-language models. Indeed, point cloud and 3D meshes typically have a lower resolution than images and the reconstructed 3D scene geometry might not project well to the underlying 2D image sequences used to compute pixel-aligned CLIP features. To address these challenges, we propose OpenNeRF which naturally operates on posed images and directly encodes the VLM features within the NeRF. This is similar in spirit to LERF, however our work shows that using pixel-wise VLM features (instead of global CLIP features) results in an overall less complex architecture without the need for additional DINO regularization. Our OpenNeRF further leverages NeRF's ability to render novel views and extract open-set VLM features from areas that are not well observed in the initial posed images. For 3D point cloud segmentation on the Replica dataset, OpenNeRF outperforms recent open-vocabulary methods such as LERF and OpenScene by at least +4.9 mIoU.
</details>


### [arXiv2404] Gaga: Group Any Gaussians via 3D-aware Memory Bank

**Authors**: Weijie Lyu, Xueting Li, Abhijit Kundu, Yi-Hsuan Tsai, Ming-Hsuan Yang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2404.07977) | [🌐 Project Page](https://www.gaga.gallery/) | [💻 Code (not yet)](https://github.com/weijielyu/Gaga)

<details span>
<summary><b>Abstract</b></summary>
We introduce Gaga, a framework that reconstructs and segments open-world 3D scenes by leveraging inconsistent 2D masks predicted by zero-shot segmentation models. Contrasted to prior 3D scene segmentation approaches that heavily rely on video object tracking, Gaga utilizes spatial information and effectively associates object masks across diverse camera poses. By eliminating the assumption of continuous view changes in training images, Gaga demonstrates robustness to variations in camera poses, particularly beneficial for sparsely sampled images, ensuring precise mask label consistency. Furthermore, Gaga accommodates 2D segmentation masks from diverse sources and demonstrates robust performance with different open-world zero-shot segmentation models, enhancing its versatility. Extensive qualitative and quantitative evaluations demonstrate that Gaga performs favorably against state-of-the-art methods, emphasizing its potential for real-world applications such as scene understanding and manipulation.
</details>


### [arXiv2407] Fast and Efficient: Mask Neural Fields for 3D Scene Segmentation

**Authors**: Zihan Gao, Lingling Li, Licheng Jiao, Fang Liu, Xu Liu, Wenping Ma, Yuwei Guo, Shuyuan Yang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2407.01220)

<details span>
<summary><b>Abstract</b></summary>
Understanding 3D scenes is a crucial challenge in computer vision research with applications spanning multiple domains. Recent advancements in distilling 2D vision-language foundation models into neural fields, like NeRF and 3DGS, enables open-vocabulary segmentation of 3D scenes from 2D multi-view images without the need for precise 3D annotations. While effective, however, the per-pixel distillation of high-dimensional CLIP features introduces ambiguity and necessitates complex regularization strategies, adding inefficiencies during training. This paper presents MaskField, which enables fast and efficient 3D open-vocabulary segmentation with neural fields under weak supervision. Unlike previous methods, MaskField distills masks rather than dense high-dimensional CLIP features. MaskFields employ neural fields as binary mask generators and supervise them with masks generated by SAM and classified by coarse CLIP features. MaskField overcomes the ambiguous object boundaries by naturally introducing SAM segmented object shapes without extra regularization during training. By circumventing the direct handling of high-dimensional CLIP features during training, MaskField is particularly compatible with explicit scene representations like 3DGS. Our extensive experiments show that MaskField not only surpasses prior state-of-the-art methods but also achieves remarkably fast convergence, outperforming previous methods with just 5 minutes of training. We hope that MaskField will inspire further exploration into how neural fields can be trained to comprehend 3D scenes from 2D models.
</details>


### [arXiv2407] Segment Any 4D Gaussians

**Authors**: Shengxiang Ji, Guanjun Wu, Jiemin Fang, Jiazhong Cen, Taoran Yi, Wenyu Liu, Qi Tian, Xinggang Wang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2407.04504) | [🌐 Project Page](https://jsxzs.github.io/sa4d/) | [💻 Code (not yet)](https://github.com/jsxzs/SA4D)

<details span>
<summary><b>Abstract</b></summary>
Modeling, understanding, and reconstructing the real world are crucial in XR/VR. Recently, 3D Gaussian Splatting (3D-GS) methods have shown remarkable success in modeling and understanding 3D scenes. Similarly, various 4D representations have demonstrated the ability to capture the dynamics of the 4D world. However, there is a dearth of research focusing on segmentation within 4D representations. In this paper, we propose Segment Any 4D Gaussians (SA4D), one of the first frameworks to segment anything in the 4D digital world based on 4D Gaussians. In SA4D, an efficient temporal identity feature field is introduced to handle Gaussian drifting, with the potential to learn precise identity features from noisy and sparse input. Additionally, a 4D segmentation refinement process is proposed to remove artifacts. Our SA4D achieves precise, high-quality segmentation within seconds in 4D Gaussians and shows the ability to remove, recolor, compose, and render high-quality anything masks. More demos are available at: this https URL.
</details>


### [ECCV24] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians

**Authors**: Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong Kim, Hoseok Do

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2407.11793) | [🌐 Project Page](https://seokhunchoi.github.io/Click-Gaussian/)

<details span>
<summary><b>Abstract</b></summary>
Interactive segmentation of 3D Gaussians opens a great opportunity for real-time manipulation of 3D scenes thanks to the real-time rendering capability of 3D Gaussian Splatting. However, the current methods suffer from time-consuming post-processing to deal with noisy segmentation output. Also, they struggle to provide detailed segmentation, which is important for fine-grained manipulation of 3D scenes. In this study, we propose Click-Gaussian, which learns distinguishable feature fields of two-level granularity, facilitating segmentation without time-consuming post-processing. We delve into challenges stemming from inconsistently learned feature fields resulting from 2D segmentation obtained independently from a 3D scene. 3D segmentation accuracy deteriorates when 2D segmentation results across the views, primary cues for 3D segmentation, are in conflict. To overcome these issues, we propose Global Feature-guided Learning (GFL). GFL constructs the clusters of global feature candidates from noisy 2D segments across the views, which smooths out noises when training the features of 3D Gaussians. Our method runs in 10 ms per click, 15 to 130 times as fast as the previous methods, while also significantly improving segmentation accuracy. Our project page is available at this https URL
</details>

<br>


## For Editing

### [NIPS22] Decomposing NeRF for Editing via Feature Field Distillation

**Authors**: Sosuke Kobayashi, Eiichi Matsumoto, Vincent Sitzmann

**Resources:** [📄 Paper](https://arxiv.org/pdf/2205.15585.pdf) | [🌐 Project Page](https://pfnet-research.github.io/distilled-feature-fields/) | [💻 Code](https://github.com/pfnet-research/distilled-feature-fields)

<details span>
<summary><b>Abstract</b></summary>
Emerging neural radiance fields (NeRF) are a promising scene representation for computer graphics, enabling high-quality 3D reconstruction and novel view synthesis from image observations. However, editing a scene represented by a NeRF is challenging, as the underlying connectionist representations such as MLPs or voxel grids are not object-centric or compositional. In particular, it has been difficult to selectively edit specific regions or objects. In this work, we tackle the problem of semantic scene decomposition of NeRFs to enable query-based local editing of the represented 3D scenes. We propose to distill the knowledge of off-the-shelf, self-supervised 2D image feature extractors such as CLIP-LSeg or DINO into a 3D feature field optimized in parallel to the radiance field. Given a user-specified query of various modalities such as text, an image patch, or a point-and-click selection, 3D feature fields semantically decompose 3D space without the need for re-training and enable us to semantically select and edit regions in the radiance field. Our experiments validate that the distilled feature fields (DFFs) can transfer recent progress in 2D vision and language foundation models to 3D scene representations, enabling convincing 3D segmentation and selective editing of emerging neural graphics representations.
</details>


### [arXiv2312] Gaussian Grouping: Segment and Edit Anything in 3D Scenes 

**Authors**: Mingqiao Ye, Martin Danelljan, Fisher Yu, Lei Ke 

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2312.00732.pdf) | [💻 Code](https://github.com/lkeab/gaussian-grouping) 

<details span>
<summary><b>Abstract</b></summary>
The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by SAM, along with introduced 3D spatial consistency regularization. Comparing to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization and scene recomposition. 
</details>


### [arXiv2312] 4D-Editor: Interactive Object-level Editing in Dynamic Neural Radiance Fields via Semantic Distillation

**Authors**: Dadong Jiang, Zhihui Ke, Xiaobo Zhou, Xidong Shi

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2310.16858.pdf) | [🌐 Project Page](https://patrickddj.github.io/4D-Editor/) | [💻 Code (not yet)]()

<details span>
<summary><b>Abstract</b></summary>
This paper targets interactive object-level editing (e.g., deletion, recoloring, transformation, composition) in dynamic scenes. Recently, some methods aiming for flexible editing static scenes represented by neural radiance field (NeRF) have shown impressive synthesis quality, while similar capabilities in time-variant dynamic scenes remain limited. To solve this problem, we propose 4D-Editor, an interactive semantic-driven editing framework, allowing editing multiple objects in a dynamic NeRF with user strokes on a single frame. We propose an extension to the original dynamic NeRF by incorporating a hybrid semantic feature distillation to maintain spatial-temporal consistency after editing. In addition, we design Recursive Selection Refinement that significantly boosts object segmentation accuracy within a dynamic NeRF to aid the editing process. Moreover, we develop Multi-view Reprojection Inpainting to fill holes caused by incomplete scene capture after editing. Extensive experiments and editing examples on real-world demonstrate that 4D-Editor achieves photo-realistic editing on dynamic NeRFs.
</details>


### [arXiv2405] TIGER: Text-Instructed 3D Gaussian Retrieval and Coherent Editing   

**Authors**: Teng Xu, Jiamin Chen, Peng Chen, Youjia Zhang, Junqing Yu, Wei Yang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2405.14455)

<details span>
<summary><b>Abstract</b></summary>
Editing objects within a scene is a critical functionality required across a broad spectrum of applications in computer vision and graphics. As 3D Gaussian Splatting (3DGS) emerges as a frontier in scene representation, the effective modification of 3D Gaussian scenes has become increasingly vital. This process entails accurately retrieve the target objects and subsequently performing modifications based on instructions. Though available in pieces, existing techniques mainly embed sparse semantics into Gaussians for retrieval, and rely on an iterative dataset update paradigm for editing, leading to over-smoothing or inconsistency issues. To this end, this paper proposes a systematic approach, namely TIGER, for coherent text-instructed 3D Gaussian retrieval and editing. In contrast to the top-down language grounding approach for 3D Gaussians, we adopt a bottom-up language aggregation strategy to generate a denser language embedded 3D Gaussians that supports open-vocabulary retrieval. To overcome the over-smoothing and inconsistency issues in editing, we propose a Coherent Score Distillation (CSD) that aggregates a 2D image editing diffusion model and a multi-view diffusion model for score distillation, producing multi-view consistent editing with much finer details. In various experiments, we demonstrate that our TIGER is able to accomplish more consistent and realistic edits than prior work.
</details>

<br>

## SLAM

### [CVPR24] SNI-SLAM: Semantic Neural Implicit SLAM

**Authors**: Siting Zhu*, Guangming Wang*, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, Hesheng Wang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2311.11016.pdf) | [💻 Code (not yet)](https://github.com/IRMVLab/SNI-SLAM)

<details span>
<summary><b>Abstract</b></summary>
We propose SNI-SLAM, a semantic SLAM system utilizing neural implicit representation, that simultaneously performs accurate semantic mapping, high-quality surface reconstruction, and robust camera tracking. In this system, we introduce hierarchical semantic representation to allow multi-level semantic comprehension for top-down structured semantic mapping of the scene. In addition, to fully utilize the correlation between multiple attributes of the environment, we integrate appearance, geometry and semantic features through cross-attention for feature collaboration. This strategy enables a more multifaceted understanding of the environment, thereby allowing SNI-SLAM to remain robust even when single attribute is defective. Then, we design an internal fusion-based decoder to obtain semantic, RGB, Truncated Signed Distance Field (TSDF) values from multi-level features for accurate decoding. Furthermore, we propose a feature loss to update the scene representation at the feature level. Compared with low-level losses such as RGB loss and depth loss, our feature loss is capable of guiding the network optimization on a higher-level. Our SNI-SLAM method demonstrates superior performance over all recent NeRF-based SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in accurate semantic segmentation and real-time semantic mapping.
</details>

### [TVCG24] NIS-SLAM: Neural Implicit Semantic RGB-D SLAM for 3D Consistent Scene Understanding

**Authors**: Hongjia Zhai, Gan Huang, Qirui Hu, Guanglin Li, Hujun Bao, Guofeng Zhang

**Resources:**  [📄 Paper](https://arxiv.org/pdf/2407.20853) | [🌐 Project Page](https://zju3dv.github.io/nis_slam/) | [💻 Code (not yet)]()

<details span>
<summary><b>Abstract</b></summary>
In recent years, the paradigm of neural implicit representations has gained substantial attention in the field of Simultaneous Localization and Mapping (SLAM). However, a notable gap exists in the existing approaches when it comes to scene understanding. In this paper, we introduce NIS-SLAM, an efficient neural implicit semantic RGB-D SLAM system, that leverages a pre-trained 2D segmentation network to learn consistent semantic representations. Specifically, for high-fidelity surface reconstruction and spatial consistent scene understanding, we combine high-frequency multi-resolution tetrahedron-based features and low-frequency positional encoding as the implicit scene representations. Besides, to address the inconsistency of 2D segmentation results from multiple views, we propose a fusion strategy that integrates the semantic probabilities from previous non-keyframes into keyframes to achieve consistent semantic learning. Furthermore, we implement a confidence-based pixel sampling and progressive optimization weight function for robust camera tracking. Extensive experimental results on various datasets show the better or more competitive performance of our system when compared to other existing neural dense implicit RGB-D SLAM approaches. Finally, we also show that our approach can be used in augmented reality applications. Project page: https://zju3dv.github.io/nis_slam.
</details>

<br>

## Resources

* [DINO](https://github.com/facebookresearch/dino)
* [DINOv2](https://github.com/facebookresearch/dinov2)
* [CLIP](https://github.com/openai/CLIP)
* [OpenCLIP](https://github.com/mlfoundations/open_clip)
* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

