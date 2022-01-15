# When Causality meets Computer Vision

The repository contains lists of papers on causality and how relevant techniques are being used to further enhance deep learning era computer vision solutions. 

The repository is organized by [Maheep Chaudhary](https://maheepchaudhary.github.io/maheep.github.io/) and [Haohan Wang](http://www.cs.cmu.edu/~haohanw/) as an effort to collect and read relevant papers and to hopefully serve the public as a collection of relevant resources. 

## Causality 
<!--- 
1.) Not able to absorb properly......... 
2.) not geting to the desired speed
-->
 

## Causality & Computer Vision

<!--- Week 1 --> 
  - [Adversarial Visual Robustness by Causal Intervention](https://arxiv.org/abs/2106.09534) 
      - <details><summary>Maheep's notes </summary>
         The paper focuses on adverserial training so as to prevent from adverserial attacks. The author use instrumental variable to achieve casual intervention.            The author proposes 2 techniques, i.e. 
         
         1) Augments the image with multiple retinoptic centres
         
         2) Encourage the model to learn causal features, rather than local confounding patterns.
         
         They propose the model to be such that max P (Y = ŷ|X = x + delta) - P(Y = ŷ|do(X = x + delta)), 
         subject to P (Y = ŷ|do(X = x + delta)) = P (Y = ŷ|do(X = x)), in other words they focus on annhilating the confounders using the retinotopic centres as the instrumental variable.
        </details>
---
<!--- ### Week 2 -->

  - [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/pdf/2002.11949.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on scene graph generation (SGG) task based on causal inference. The author use Total Direct Effect for an unbiased SGG. The author proposes the technique, i.e. 
         
         1) To take remove the context bias, the author compares it with the counterfactual scene, where visual features are wiped out(containing no objects). 
         
         The author argues that the true label is influenced by Image(whole content of the image) and context(individual objects, the model make a bias that the object is only to sit or stand for and make a bias for it) as confounders, whereas we only need the Content(object pairs) to make the true prediction. 
         The author proposes the TDE = y_e - y_e(x_bar,z_e), the first term denote the logits of the image when there is no intervention, the latter term signifies the logit when content(object pairs) are removed from the image, therfore giving the total effect of content and removing other effect of confounders.    
        </details>
        
   - [Counterfactual Visual Explanations](https://arxiv.org/pdf/1904.07451.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on Counterfactual Visual Explanations. The author ask a very signifiant question while developing the technique proposed in the paper, i.e. how I could change such that the system would output a different specified class c'. To do this, the author proposes the defined technique: - 
         
         1) He selects ‘distractor’ image I' that the system predicts as class c' and identify spatial regions in I and I' such that replacing the identified region in I with the identified region in I' would push the system towards classifying I as c'. 
        
        The author proposes the implementation by the equation: <br>
        `f(I*) = (1-a)*f(I) + a*P(f(I'))` <br> 
        where <br> 
        `I*` represents the image made using the `I` and `I'`
        `*` represents the Hamdard product. <br>
        `f(.)` represents the spatial feature extractor <br>
        `P(f(.))` represents a permutation matrix that rearranges the spatial cells of `f(I')` to align with spatial cells of `f(I)`
      
        The author implements it using the two greedy sequential relaxations – first, an exhaustive search approach keeping a and P binary and second, a continuous relaxation of a and P that replaces search with an optimization. 
        </details>

   - [Counterfactual Vision and Language Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Abbasnejad_Counterfactual_Vision_and_Language_Learning_CVPR_2020_paper.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on VQA models using the counterfactual intervention to make it robust. They ask a crucial question, i.e. “what would be the minimum alteration to the question or image that could change the answer”. The author uses the observational data as well as the counterfactual data to predict the answer. To do this, the author proposes the defined technique: - 
         
         1) The author replaces the embedding of the question or image using another question or image so as to predict the correct answer and minimize counterfactual loss. 
        </details>
        
   - [Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler](https://arxiv.org/pdf/1911.07308.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on Vision-and-Language Navigation (VLN). The author combine the adversarial training with counterfactual conditions to guide models that might lead to robust model. To do this, the author proposes the defined techniques: - 
         
         1) The author APS, i.e. adversarial path sampler which samples batch of paths P after augmenting them and reconstruct instructions I using Speaker. With the pairs of (P,I), so as to maximize the navigation loss L_NAV. 
         2) The NAV, i.e. navigation model trains so as to minimize the L_Nav making the whole process more robust and increasing the performance. 
         
        The APS samples the path based on the visual features v_t which are obtained using the attention on the feature space f_t and history h_t-1 and previous action taken a_t-1 to output the path using the predicted a_t and the features f_t.
        </details>


   - [Beyond Trivial Counterfactual Explanations with Diverse Valuable Explanations](https://arxiv.org/pdf/2103.10226.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper system DiVE in a world where the works are produced to get features which might change the output of the image and learns a perturbation in a disentangled latent space that is constrained using a diversity-enforcing loss to uncover multiple valuable explanations about the model’s prediction. The author proposes these techniques to get the no-trivial explanations and making the model more diversified, sparse and valid: - 
         
         1) DiVE uses an encoder, a decoder, and a fixed-weight ML model.
         2) Encoder and Decoder are trained in an unsupervised manner to approximate the data distribution on which the ML model was trained. 
         3) They optimize a set of vectors E_i to perturb the latent representation z generated by the trained encoder.
          
        The author proposes 3 main losses: 
        `Counterfatual loss` : It identifies a change of latent attributes that will cause the ML model f to change it’s prediction.
        `Proximity loss` : The goal of this loss function is to constrain the reconstruction produced by the decoder to be similar in appearance and attributes as the input, therfore making the model sparse. 
        `Diversity loss` : This loss prevents the multiple explanations of the model from being identical and reconstructs different images modifing the different spurious correlations and explaing through them.
        The model uses the beta-TCVAE to obtain a disentangled latent representation which leads to more proximal and sparse explanations and also Fisher information matrix of its latent space to focus its search on the less influential factors of variation of the ML model as it defines the scores of the influential latent factors of Z. This mechanism enables the discovery of spurious correlations learned by the ML model.
        </details>       
         
        
        
   - [SCOUT: Self-aware Discriminant Counterfactual Explanations](https://arxiv.org/pdf/2004.07769.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes to connect attributive explanations, which are based on a single heat map, to counterfactual explanations, which seek to identify regions where it is easy to discriminate between prediction and counter class. They also segments the region which discriminates between the two classes of a class image. 
        
        The author implements using a network by giving a query image x of class y , a user-chosen counter class y' != y, a predictor h(x), and a confidence predictor s(x), x is then forwarded to get the F_h(x) and F_s(x). From F_h(x) we predict h_y(x) and h_y'(x) which are then combined with the original F_h(x) to produce the A(x, y) and A(x, y') to get the activation tensors and they are then combined with A(x, s(x)) to get the segmented region of the image which is discriminative of the counter class. 
        </details>         
     
     
                 
   - [Born Identity Network: Multi-way Counterfactual Map Generation to Explain a Classifier’s Decision](https://arxiv.org/pdf/2011.10381.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes a system BIN that is used to produce counterfactual maps as a step towards counterfactual reasoning, which is a process of producing hypothetical realities given observations. The system proposes techniques: - 
         
         1) The author proposes Counterfactual Map Generator (CMG), which consists of an encoder E , a generator G , and a discriminator D . First, the network design of the encoder E and the generator G is a variation of U-Net with a tiled target label concatenated to the skip connections. This generator design enables the generation to synthesize target conditioned maps such that multi-way counterfactual reasoning is possible.  
         2) The another main technique porposes is the Target Attribution Network(TAN) the objective of the TAN is to guide the generator to produce counterfactual maps that transform an input sample to be classified as a target class. It is a complementary to CMG.

        The author proposes 3 main losses: 
        `Counterfatual Map loss` : The counterfactual map loss limits the values of the counterfactual map to grow as done by proximity loss in DiVE.
        `Adverserial loss` : It is an objective function reatained due to its stability during adversarial training. 
        `Cycle Consistency loss` : The cycle consistency loss is used for producing better multi-way counterfactual maps. However, since the discriminator only classifies the real or fake samples, it does not have the ability to guide the generator to produce multi-way counterfactual maps.
        </details>

---        
<!--- Week 3 -->

   - [Introspective Distillation for Robust Question Answering](https://arxiv.org/pdf/2111.01026.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on the fact that the present day systems to make more genralized on OOD(out-of-distribution) they sacrifice their performance on the ID(in-distribution) data. To achieve a better performance in real-world the system need to have accuracy on both the distributions to be good. Keeping this in mind the author proposes: - 
         
         1) The author proposes to have a causal feature to teach the model both about the OOD and ID data points and take into account the `P_OOD` and `P_ID`, i.e. the predictions of ID and OOD.  
         2) Based on the above predictions the it can be easily introspected that which one of the distributions is the model exploiting more and based on it they produce the second barnch of the model that scores for `S_ID` and `S_OOD` that are based on the equation `S_ID = 1/XE(P_GT, P_ID)`, where `XE` is the cross entropy loss. further these scores are used to compute weights `W_ID` and `W_OOD`, i.e. `W_OOD = S_OOD/(S_OOD + S_ID)` to train the model to blend the knowledge from both the OOD and ID data points. 
         3) The model is then distilled using the knowledge distillation manner, i.e. `L = KL(P_T, P_S)`, where `P_T` is the prediction of the teacher model and the `P_S` is the prediction of the student model. 
        </details>
        
   - [Counterfactual Explanation and Causal Inference In Service of Robustness in Robot Control](https://arxiv.org/pdf/2009.08856.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on the generating the features using counterfactual mechanism so as to make the model robust. The author proposes to generate the features which are minimal and realistic in an image so as to make it as close as the training image to make the model work correctly making the model robust to adverserial attacks, therfore robust. The generator has two main components, a discriminator which forces the generator to generate the features that are similar to the output class and the modification has to be as small as possible.   <br><br>

        The additonal component in the model is the predictor takes the modified image and produces real-world output. The implementation of it in mathematics looks like: <br>
        `min d_g(x, x') + d_c(C(x'), t_c)`, where d_g is the distance b/w the modified and original image, d_c is the distance b/w the class space and C is the predictor that x' belongs to t_c class. <br>
        The loss defines as: `total_loss = (1-alpha)*L_g(x, x') + (alpha)*L_c(x, t_c)`, where L_c is the loss `x` belongs to `t_c` class   

        </details>

   - [Counterfactual Explanation Based on Gradual Construction for Deep Networks](https://arxiv.org/pdf/2008.01897.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on gradually construct an explanation by iterating over masking and composition steps, where the masking step aims to select the important feature from the input data to be classified as target label. The compostition step aims to optimize the previously selected features by perturbating them so as to prodice the target class. <br><br>

        The proposed also focuses on 2 things, i.e. Explainability and Minimality. while implementing the techniue the authors observe the target class which were being generated were getting much perturbated so as to come under asverserial attack and therfore they propose the logit space of x' to belong to the space of training data as follows:   
        `argmin(sigma(f_k'(x') - (1/N)*sigma(f_k'(X_i,c_t))) + lambda(X' - X))` <br>
        where `f'` gives the logits for class k, `X_i,c_t` represents the i-th training data that is classified into c_k class and the N is the number of modifications. 
        </details>
        
   - [CoCoX: Generating Conceptual and Counterfactual Explanations via Fault-Lines](https://ojs.aaai.org/index.php/AAAI/article/view/5643/5499)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses a model for explaining decisions made by a deep convolutional neural network (CNN) fault-lines that defines the main features from which the humans deifferentiate the two similar classes. The author introduces 2 concepts: PFT and NFT, PFT are those xoncepts to be added to input image to change model prediction and for NFT it subtracts, whereas the xconcepts are those semantic features that are main features extracted by CNN and from which fault-lines are made by selecting from them.<br><br>

        The proposed model is implemented by taking the CNN captured richer semantic aspect and construct xconcepts by making use of feature maps from the last convolution layer. Every feature map is treated as an instance of an xconcept and obtain its localization map using the Grad-CAM and are spatially pooled to get important weights, based on that top p pixels are selected and are clustered using K-means. The selection is done using the TCAV tecnique. 

        !['Algorithm'](images/1.png)
        </details>
        
   - [CX-ToM: Counterfactual Explanations with Theory-of-Mind for Enhancing Human Trust in Image Recognition Models](https://arxiv.org/pdf/2109.01401.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper is kind of an extension of the above paper(CoCoX), i.e. it also uses fault-lines for explainability but states a dialogue between a user and the machine. The model is made by using the fault-lines and the Theory of Mind(ToM). <br><br>
        The proposed is implemented by taking an image and the same image is blurred and given to a person, then the machine take out the crucial features by thinking what the person may have understood and what is the information it should provide. The person is given more images and then the missing parts are told to be predicted after the dialogue, if the person is able to predict the parts that it was missing before then the machine gets a positive reward and functions in a RL training technique way.  

        !['Algorithm'](images/2.png)
        </details>
        
   - [DeDUCE: Generating Counterfactual Explanations At Scale](https://arxiv.org/pdf/2111.15639.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focues to detect the erroneous behaviour of the models using counterfatctual as when an image classifier outputs a wrong class label, it can be helpful to see what changes in the image would lead to a correct classification. In these cases the counterfactual acrs as the closest alternative that changes the prediction and we also learn about the decision boundary.<br><br>
        The proposed model is implemented by identifying the Epistemic uncertainity, i.e. the useful features using the Gaussian Mixture Model and therfore only the target class density is increased. The next step would be to change the prediction using a subtle change therefore the most slaient pixel, identified usign the gradient are changed.  

        !['Algorithm'](images/3.png)
        </details>     
        
   - [Designing Counterfactual Generators using Deep Model Inversion](https://arxiv.org/pdf/2109.14274.pdf)
      - <details><summary>Maheep Notes</summary>
        The paper focues on the scenario when the we have access only to the trained deep classifier and not the actual training data. The paper proposes a goal to develop a deep inversion approach to generate counterfactual explanations. The paper propses methods to preserve metrics for semantic preservation using the different methods such as ISO and LSO. The author also focuses on manifold consistency for the counterfactual image using the Deep Image Prior model. -

        `argmin(lambda_1*sigma_on_l(layer_l(x'), layer_l(x)) + lambda_2*L_mc(x';F) + lambda_3*L_cf(F(x'), y'))` 
        <br>
        where, <br>
        `layer_l` :The differentiable layer "l" of the neural network, it is basically used for semantic preservation. <br>
        `L_mc`: It penlaizes x' whcih do not lie near the manifold. L_mc can be Deterministic Uncertainty Quantification (DUQ).<br>
        `L_fc`: It ensures that the prediction for the counterfactual matches the desired target
        </details>
        
        
   - [ECINN: Efficient Counterfactuals from Invertible Neural Networks](https://arxiv.org/pdf/2103.13701.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper utilizes the generative capacities of invertible neural networks for image classification to generate counterfactual examples efficiently. The main advantage of this network is that it is fast and invertible, i.e. it has full information preservation between input and output layers, where the other networks are surjective in nature, therfore also making the evaluation easy. The network claims to change only class-dependent features while ignoring the class-independence features succesfully. This happens as the INNs have the property that thier latent spaces are semantically organized. When many latent representations of samples from the same class are averaged, then class-independent information like background and object orientation will cancel out and leaves just class-dependent information<br>

        `x' = f_inv(f(x) + alpha*delta_x)` 
        <br>
        where, <br>
        `x'` :Counterfactual image. <br>
        `f`: INN and therfore `f_inv` is the inverse of `f`.<br>
        `delta_x`: the infoprmation to be added to convert the latent space of image to that of counterfactual image.<br>
        `||z + alpha_0*delta_x- µ_p || = ||z + alpha_0*delta_x - µ_q ||` where the z + alpha_0*delta_x is the line separating the two classes and µ_q and µ_q are the mean distance from line. Therefore <br>
        `alpha = alpha_0 + 4/5*(1-alpha_0)`  
        </details>
        
   - [EXPLAINABLE IMAGE CLASSIFICATION WITH EVIDENCE COUNTERFACTUAL](https://arxiv.org/pdf/2004.07511.pdf)
      - <details><summary>Maheep's Notes</summary>
        The author proposes a SDEC model that searches a small set of segments that, in case of removal, alters the classification<br>
        The image is segemented with l segments and then the technique is implemented by using the best-first search avoid a complete search through all possible segment combinations. The best-first is each time selected based on the highest reduction in predicted class score. It continues until one or more same-sized explanations are found after an expansion loop. An additional local search can be performed by considering all possible subsets of the obtained explanation. If a subset leads to a class change after removal, the smallest set is taken as final explanation. When different subsets of equal size lead to a class change, the one with the highest reduction in predicted class score can be selected.

        !['Algorithm'](images/4.png)
        </details>      
        
        
   - [Explaining Visual Models by Causal Attribution](https://arxiv.org/pdf/1909.08891.pdf)
      - <details><summary>Maheep Notes</summary>
        The paper focuses on the facts that there are limitations of current Conditional Image Generators for Counterfactual Generation and also proposes a new explanation technique for visual models based on latent factors. <br>
        The paper is implemented using the Distribution Causal Graph(DCG) where the causal graph is made but the nodes is represented the MLP, i.e. 
        `logP(X = (x1,x2,x3...xn)) = sigma(log(P(X = xi|theta_i)))` and the Counterfactual Image Generator which translate the latent factor into the image using the original image as anchor while genrating it which is done using Fader Networks which adds a critic in the latent space and AttGAN adds the critic in the actual output. 
        </details>      

   - [Explaining the Black-box Smoothly-A Counterfactual Approach](https://arxiv.org/pdf/2101.04230.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on explaining the outcome of medical imaging by gradually exaggerating the semantic effect of the given outcome label and also show a counterfactual image by introducing the perturbations to the query image that gradually changes the posterior probability from its original class to its negation. The explanation therfore consist of 3 properties:<br>
        1.) **Data Consistency**: The resembalance of the generated and orignal data should be same. Therefore cGAN id introduced with a loss as
        
        `L_cgan = log(P_data(x)/q(x)) + log(P_data(c|x)/q(c|x))` 
        
        where P_data(x) is the data distribtion and learned distribution q(x), whreas P_data(c|x)/q(c|x) = r(c|x) is the ratio of the generated image and the condition.  <br>
        2.) **Classification model consistency**: The generated image should give desired output. Therefore the condition-aware loss is introduced, i.e. 
        `L := r(c|x) + D_KL (f(x')||f (x) + delta),`, where f(x') is the output of classifier of the counterfactual image is varied only by delta amount when added to original image logit. They take delta as a knob to regularize the genration of counterfactual image. <br>
        3.) **Context-aware self-consistency**: To be self-consistent, the explanation function should satisfy three criteria <br>
        > (a) Reconstructing the input image by setting = 0 should return the input image, i.e., G(x, 0) = x. <br> 
        > (b) Applying a reverse perturbation on the explanation image x should recover x. <br> 

        To mitigate this conditions the author propose an identity loss. The author argues that there is a chance that the GAN may ignore small or uncommon details therfore the images are compared using semantic segemntation with object detection combined in identity loss. The identity loss is :
        L_identity = L_rec(x, G(x, 0))+ L_rec(x, G(G(x,delta), -delta))
        </details>      
                
        
   - [Explaining the Behavior of Black-Box Prediction Algorithms with Causal  Learning](https://arxiv.org/pdf/2006.02482.pdf)
      - <details><summary>Noting........</summary>
        The paper 
        </details>         
        
        
   - [Explaining Classifiers with Causal Concept Effect (CaCE)](https://arxiv.org/pdf/1907.07165.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes a system CaCE, which focuses on confounding of concepts, i.e higher level unit than low level, individual input features such as pixels by intervening on concepts by taking an important assumption that intervention happens atomically. The effect is taken as 

        `Effect = E(F(I)|do(C = 1)) - E(F(I)|do(C = 0))` where F gives output on image I and C is the concept. This can be done at scale by intervening for a lot of values in a concept and find the spurious corrlation. But due to the insufficient knowlegde of the Causal Graph teh author porposes a VAE which can calculate the precise CaCE by by generating counterfactual image by just changing a concept and hence computing the difference between the prediction score.  
        </details>         
                
   - [Fast Real-time Counterfactual Explanations](https://arxiv.org/pdf/2007.05684.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes a transformer is trained as a residual generator conditional on a classifier constrained under a proposal perturbation loss which maintains the content information of the query image, but just the class-specific semantic information is changed. The technique is implemented as : <br>

        1.) **Adverserial loss**: It measures whether the generated image is indistinguishable from the real world images <br>
        2.) **Domain classification loss**: It is used to render the generate image x + G(x,y') conditional on y'. `L = E[-log(D(y'|x + G(x,y')))]` where G(x, y') is the perterbuation introduced by generator to convert image from x to x' <br>
        3.) **Reconstruction loss**: The Loss focuses to have generator work propoerly so as to produce the image need to be produced as defined by the loss. `L = E[x - (x + G(x,y') + G(x + G(x,y'), y))]`
        4.) **Explanation loss**: This is to gurantee that the generated fake image produced belongs to the distribution of H. `L = E[-logH(y'|x + G(x,y'))]`        
        5.) **Perturbation loss**: To have the perturbation as small as possible it is introduced. `L = E[G(x,y') + G(x + G(x,y'),y)]`
        <br>
        All these 5 losses are added to make the final loss with different weights.
        </details>         
        

   - [GENERATIVE_COUNTERFACTUAL_INTROSPECTION_FOR_EXPLAINABLE_DEEP_LEARNING](https://arxiv.org/pdf/1907.03077.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper propose to generate counterfactual using the Generative Counterfactual Explanation not by replacing a patch of the original image with something but by generating a counterfactual image by replacing minimal attributes uinchanged, i.e. A = {a1, a2, a3, a4, a5....an}. It is implemented by: - <br>

        `min(lambda*loss(I(A')) + ||I - I(A'))`, where loss is cross-entropy for predicting image I(A') to label c'.

        </details>        
        
  - [Generative_Counterfactuals_for_Neural_Networks_via_Attribute_Informed_Perturbations](https://arxiv.org/pdf/2101.06930.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focues on generating counterfactuals for raw data instances (i.e., text and image) is still in the early stage due to its challenges on high data dimensionality, unsemantic raw features and also in scenario when the effictive counterfactual for certain label are not guranteed, therfore the author proposes Attribute-Informed-Perturbation(AIP) which convert raw features are embedded as low-dimension and data attributes are modeled as joint latent features. To make this process optimized it has two losses: Reconstruction_loss(used to guarantee the quality of the raw feature) + Discrimination loss,(ensure the correct the attribute embedding) i.e.  

        `min(E[sigma_for_diff_attributes*(-a*log(D(x')) - (1-a)*(1-D(x)))]) + E[||x - x'||]` where D(x') generates attributes for counterfactual image.<br> To generate the counterfactual 2 losses are produced,one ensures that the perturbed image has the desired label and the second one ensures that the perturbation is minimal as possible, i.e. <br> `L_gen = Cross_entropy(F(G(z, a)), y) + alpha*L(z,a,z_0, a_0)`<br>
        The L(z,a,z0,a0) is the l2 norm b/w the attribute and the latent space.
        </details>        
        
   - [Question-Conditioned Counterfactual Image Generation for VQA](https://arxiv.org/pdf/1911.06352.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper on generating the counterfactual images for VQA, s.t. <br>
        i.) the VQA model outputs a different answer<br>
        ii.) the new image is minimally different from the original <br>
        iii) the new image is realistic <br>
        The author uses a LingUNet model for this and proposes three losses to make the perfect. <br>
        1.) Negated cross entropy for VQA model. <br> 
        2.) l2 loss b/w the generated image and the original image.
        3.) Discriminator that penalizes unrealistic images.  
        </details>           
        

   - [FINDING AND FIXING SPURIOUS PATTERNS WITH EXPLANATIONS](https://arxiv.org/pdf/2106.02112.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes an augmeting technique taht resamples the images in such a way to remove the spurious pattern in them, therfore they introduce their framework Spurious Pattern Identification and REpair(SPIRE). They view the dataset as Both, Just Main, Just Spurious, and Neither. SPIRE measures this probability for all (Main, Spurious) pairs, where Main and Spurious are different, and then sorts this list to find the pairs that represent the strongest patterns. After finding the pattern the dataset is redistributes as: <br>

        `P(Spurious | Main) = P(Spurious | not Main) = 0.5`<br>
        The second step consist of minimizing the potential for new SPs by setting the <br>`P(Main|Artifact) = 0.5)`. <br>
        SPIRE moves images from {Both, Neither} to {Just Main, Just Spurious} if p > 0.5, i.e. p = P(Main|Spurious) but if p < 0.5 then SPIRE moves images from {Just Main, Just Spurious} to {Both, Neither}. 
        </details>  

   - [Contrastive_Counterfactual_Visual_Explanations_With_Overdetermination](https://arxiv.org/pdf/2106.14556.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes a system CLEAR Image that explains an image’s classification probability by contrasting the image with a corresponding image generated automatically via adversarial learning. It also provides an event with a label of "*overdetermination*", which is given when the model is more than sure that the label is something. CLEAR Image segments x into different segments S = {s1 ,...,sn } and then applies the same segmentation to x' creating S' = {s'1,...., s'n}. CLEAR Image determines the contributions that different subsets of S make to y by substituting with the corresponding segments of S'. This is impelmeted by: <br>
        A counterfactual image is generated by GAN which is then segmented and those segments by a certian threshold replace the segment in the original image and therfore we get many perturbed images. Each perturbed image is then passed through the model m to identify the classification probability of all the classes and therfore the significance of every segment is obtained that is contributing in the layer. If the
        </details> 

   - [Training_calibration‐based_counterfactual_explainers_for_deep_learning](https://www.nature.com/articles/s41598-021-04529-5)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes TraCE for deep medical imaging that trained using callibaration-technique to handle the problem of counterfactual explanation, particularly when the model's prediciton are not well-callibrated due to which it produces irrelevant feature manipulation. The system is implemeted using the 3 methods, i.e. <br>
        (1.) an auto-encoding convolutional neural network to construct a low-dimensional, continuous latent space for the training data <br>
        (2.) a predictive model that takes as input the latent representations and outputs the desired target attribute along with its prediction uncertainty<br>
        (3.) a counterfactual optimization strategy that uses an uncertainty-based calibration objective to reliably elucidate the intricate relationships between image signatures and the target attribute.<br>
        TraCE works on the following metrics to evaluate the counterfactual images, i.e. <br>

        **Validity**: ratio of the counterfactuals that actually have the desired target attribute to the total number of counterfactuals  
        The confidence of the **image** and **sparsity**, i.e. ratio of number of pixels altered to total no of pixels. Th eother 2 metrcs are **proximity**, i.e. average l2 distance of each counterfactual to the K-nearest training samples in the latent space and **Realism score** so as to have the generated image is close to the true data manifold.<br>
        TraCE reveals attribute relationships by generating counterfactual image using the different attribute like age "A" and diagnosis predictor "D". <br>
        `delta_A_x = x - x_a'` ; `delta_D_x = x - x_d'` <br>
        The x_a' is the counterfactual image on the basis for age and same for x_d'. <br>
        `x' = x + delta_A_x + delta_D_x` and hence atlast we evaluate the sensitivity of a feature by `F_d(x') - F_d(x_d')`, i.e. F_d is the classifier of diagnosis. <br>

        </details>  

   - [Generating Natural Counterfactual Visual Explanations](https://www.ijcai.org/proceedings/2020/0742.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes a counterfactual visual explainer that look for counterfactual features belonging to class B that do not exist in class A. They use each counterfactual feature to replace the corresponding class A feature and output a counterfactual text. The counterfactual text contains the B-type features of one part and the A-type features of the remaining parts. Then they use a text-to-image GAN model and the counterfactual text to generate a counterfactual image. They generate the images using the AttGAN and StackGAN and they take the image using the function. <br>
        `log(P(B)/P(A))` where P(.) is the classifier probability of a class for obtaining the highest-scoring counterfactual image. 
        </details> 

   - [On Causally Disentangled Representations](https://arxiv.org/pdf/2112.05746.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focuses on causal disentanglement that focus on disentangle factors of variation and therefore proposes two new metrics to study causal disentanglement and one dataset named CANDLE. Generative factors G is said to be disentangled only if they are influenced by their parents and not confounders. The system is implemented as: <br> <br>
        A latent model M (e,g, pX ) with an encoder e, generator g and a data distribution pX , assumes a prior p(Z) on the latent space, and a generator g is parametrized as p(X|Z), then posterior p(Z|X) is approzimated using a variational distribution q (Z|X) parametrized by another deep neural network, called the encoder e. Therefore we obtain a z for every g and acts as a proxy for it. <br>
        1.) **Unconfoundess metric**: If a model is able to map each Gi to a unique ZI ,the learned latent space Z is unconfounded and hence the property is known as unconfoundedness. <br>
        2.)**Counterfactual Generativeness**: a counterfactual instance of x w.r.t. generative factor Gi , x'(i.e., the counterfactual of x with change in only Gi) can be generated by intervening on the latents of x corresponding to Gi , ZIx and any change in the latent dimensions of Z that are x not responsible for generating G i , i.e. Z\I, should have no influence on the generated counterfactual instance x' w.r.t. generative factor Gi. It can be computed using the Avergae Causal Effect(ACE).  
        </details>  

   - [INTERPRETABILITY_THROUGH_INVERTIBILITY_A_DEEP_CONVOLUTIONAL_NETWORK](https://openreview.net/pdf?id=8YFhXYe1Ps)
      - <details><summary>Maheep's Notes</summary>
        The paper proposes a model that generates meaningful, faithful, and ideal counterfactuals. Using PCA on the classifier’s input, we can also create “isofactuals”, i.e. image interpolations with the same outcome but visually meaningful different features. The author argues that a system should provide power to the users to discover hypotheses in the input space themselves with faithful counterfactuals that are ideal. They claim that it could be easily done by combining an invertible deep neural network z = phi(x) with a linear classifier y = wT*phi(x) + b. They generate a counterfatual by altering a feature representation of x along the direction of weight vector, i.e. <br>

        `z' = z + alpha*w` where `x' = phi_inverse(z + alpha*w)`. Any change orthogonal to w will create an “isofactual. To show that their counterfactuals are ideal, therfore they verify that no property unrelated to the prediction is changed. Unrealted properties = e(x), `e(x) = vT*z`, where v is orthogonal to w. `e(x') = vT*(z + alpha* w) = vT*z = e(x)`. To measure the difference between the counterfactual and image intermediate feature map h, i.e. `m = |delta_h|*cos(angle(delta_h, h))` for every location of intermediate feature map. 
        </details>  

   - [Model-Based Counterfactual Synthesizer for Interpretation](https://arxiv.org/pdf/2106.08971.pdf)
      - <details><summary>Maheep's Notes</summary>
        The paper focues on eridicating the algorithm-based counterfactual generators which makes them ineffcient for sample generation, because each new query necessitates solving one specific optimization problem at one time and propose Model-based Counterfactual Synthesizer. Existing frameworks mostly assume the same counterfactual universe for different queries. The present methods do not consider the causal dependence among attributes to account for counterfactual feasibility. To take into account the counterfactual universe for rare queries, they novelly employ the umbrella sampling technique.
        </details>  

   - [The Intriguing Relation Between Counterfactual Explanations and Adversarial Examples](https://arxiv.org/pdf/2009.05487.pdf)
      - <details><summary>Noting.....</summary>
        The paper   
        </details>  

   - [Discriminative Attribution from Counterfactuals](https://arxiv.org/pdf/2109.13412.pdff)
      - <details><summary>Noting.....</summary>
        The paper   
        </details>  

   - [Causal Interventional Training for Image Recognition](https://ieeexplore.ieee.org/document/9656623)
      - <details><summary>Noting.....</summary>
        The paper   
        </details>  

