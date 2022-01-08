# When Causality meets Computer Vision

The repository contains lists of papers on causality and how relevant techniques are being used to further enhance deep learning era computer vision solutions. 

The repository is organized by Maheep Chaudhary and [Haohan Wang](http://www.cs.cmu.edu/~haohanw/) as an effort to collect and read relevant papers and to hopefully serve the public as a collection of relevant resources. 

## Causality 

<!--- Should I replace the arxiv link with github paper link? -->
 
## Causality & Computer Vision

### Week 1
  - [Adversarial Visual Robustness by Causal Intervention](https://arxiv.org/abs/2106.09534) 
      - <details><summary>Maheep's notes </summary>
         The paper focuses on adverserial training so as to prevent from adverserial attacks. The author use instrumental variable to achieve casual intervention.            The author proposes 2 techniques, i.e. 
         
         1) Augments the image with multiple retinoptic centres
         
         2) Encourage the model to learn causal features, rather than local confounding patterns.
         
         They propose the model to be such that max P (Y = ŷ|X = x + delta) - P(Y = ŷ|do(X = x + delta)), 
         subject to P (Y = ŷ|do(X = x + delta)) = P (Y = ŷ|do(X = x)), in other words they focus on annhilating the confounders using the retinotopic centres as the instrumental variable.
        </details>

### Week 2

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
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
