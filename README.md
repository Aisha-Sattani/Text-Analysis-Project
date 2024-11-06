# Project Title: Text Analysis and Generation with Project Gutenberg Data

## Project Overview
In this project, I used two texts from Project Gutenberg to explore text analysis techniques and implement a text generation tool. The texts used were *"The Food of the Gods: A Popular Account of Cocoa"* by Brandon Head and *"Foods That Will Win The War and How to Cook Them"* by C. Houston Goudiss and Alberta M. Goudiss. By applying methods such as word frequency counting, sentiment analysis, cosine similarity, and Markov chains, my goal was to extract insights from the books and compare themes, while also generating synthetic text in the style of each book.

## Books Used
1. **"The Food of the Gods: A Popular Account of Cocoa" by Brandon Head**  
   - *Overview*: This book discusses the history, cultivation, and cultural impact of cocoa and chocolate. It also emphasizes cocoa’s health benefits and includes practical recipes.
   - *Common Words and Themes*: Words like “chocolate,” “sugar,” “milk,” and “add” appeared frequently, highlighting the book's focus on chocolate recipes and preparation methods.
   - *Sentiment Analysis*: The tone is positive, reflecting the nutritional and cultural appreciation of cocoa. Sentences such as “a perfect food, as wholesome as delicious” convey enthusiasm for cocoa’s benefits.
   - *Markov Chain Example*: Generated sentences like “Place dish on a buttered pan, cool slightly, and add to the top of cake with the fork…” replicate the instructional style found in the book's recipes.

2. **"Foods That Will Win The War and How to Cook Them" by C. Houston Goudiss and Alberta M. Goudiss**  
   - *Overview*: Written during WWI, this book promotes economical cooking and food choices to support the war effort, focusing on accessible ingredients and minimal waste.
   - *Cosine Similarity*: The similarity score between the two books was 0.7755, indicating overlapping themes of practical cooking and ingredient-focused guidance.

## Implementation
The system is built with Python and utilizes several libraries for different analysis techniques:
1. **Word Frequency Analysis**: A histogram was created to count word occurrences, filtering out common stop words. This provided insight into the thematic focus of each book.
2. **Sentiment Analysis**: NLTK’s VADER sentiment analyzer assessed sentence-level sentiment, highlighting the emotional tone in each book’s content.
3. **Cosine Similarity and Clustering**: Cosine similarity allowed for comparing the two books to see content overlap, and MDS clustering visualized the similarity between them.
4. **Text Generation (Markov Chains)**: Markov chains were used to generate text that mimics each book’s style, producing coherent sentences based on common vocabulary and structure.

A notable design choice was using NLTK’s predefined stopwords, balancing simplicity and thematic accuracy. GenAI tools helped refine the clustering algorithm and guided me in optimizing the sentiment analysis setup.

## Results
The project achieved the following:
- **Text Analysis**: High-frequency words highlighted each book's central themes. For example, “chocolate” and “sugar” were prominent in *The Food of the Gods*, while *Foods That Will Win The War* emphasized essential ingredients like “water” and “cup.”
- **Sentiment Analysis**: Sentiment scores reflected the practical and positive tone of the book, with phrases like “highly nourishing” underscoring the value placed on food quality in both texts.
- **Cosine Similarity**: With a similarity score of 0.7755, the analysis revealed shared themes of practical recipes and ingredient focus, despite the books' differing contexts.
- **Clustering Visualization**: The MDS plot visually demonstrated content similarity between the two books.
  ![Graph](https://github.com/user-attachments/assets/c3b29bc3-28fc-4398-b5f6-d5d5d72eda04)
- **Text Generation**: Markov chains generated text consistent with each book’s style, producing recipe-like instructions for *The Food of the Gods* and practical cooking tips in the style of *Foods That Will Win The War*.

## Reflection
This project was both challenging and insightful. Testing each analysis component individually helped refine the results. From a learning perspective, I realized the versatility of text analysis in understanding themes, sentiment, and content generation. GenAI tools were instrumental in improving the clustering approach and cosine similarity graph. Going forward, I aim to explore more advanced sentiment models. Having a clearer understanding of cosine similarity and clustering at the start would have streamlined my project.  This project not only improved my technical skills but also deepened my appreciation for thematic analysis and synthetic text generation.
