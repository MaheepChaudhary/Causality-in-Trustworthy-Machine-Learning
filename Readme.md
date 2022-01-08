# When Causality meets Computer Vision

The repository contains lists of papers on causality and how relevant techniques are being used to further enhance deep learning era computer vision solutions. 

The repository is organized by Maheep Chaudhary and [Haohan Wang](http://www.cs.cmu.edu/~haohanw/) as an effort to collect and read relevant papers and to hopefully serve the public as a collection of relevant resources. 

## Causality 

<!--- Should I replace the arxiv link with github paper link? -->
 
## Causality & Computer Vision
  - [Adversarial Visual Robustness by Causal Intervention](https://arxiv.org/abs/2106.09534) 
      - <details><summary>Maheep's notes </summary>
         The paper focuses on adverserial training so as to prevent from adverserial attacks. The author use instrumental variable to achieve casual intervention.            The author proposes 2 techniques, i.e. 
         
         1) Augments the image with multiple retinoptic centres
         
         2) Encourage the model to learn causal features, rather than local confounding patterns.
         
         They propose the model to be such that max P (Y = ŷ|X = x + delta) - P(Y = ŷ|do(X = x + delta)), 
         subject to P (Y = ŷ|do(X = x + delta)) = P (Y = ŷ|do(X = x)), in other words they focus on annhilating the confounders using the retinotopic centres as the instrumental variable.
        </details>

<!--- Week 2 -->

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
        The paper focuses on Vision-and-Language Navigation (VLN) To do this, the author proposes the defined technique: - 
         
         1) The author replaces the embedding of the question or image using another question or image so as to predict the correct answer and minimize counterfactual loss. 

        </details>
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
