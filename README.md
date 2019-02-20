# SupportVectorMachine

The R script in this repository classifies text using a support vector machine model. 
It it is assumed that training data and potential live data to classify is present in tab-separated files with the following columns and column names: "Class", "ID", "Text". Sample files are available in this repository. 

## getTextClass function

Call the getTextClass function and indicate training data and/or live data. 

```r
# Text classification with Support Vector Machines

# Packages
library(tm)
library(RTextTools)
library(e1071)

getTextClass <- function(trainingData = NULL, 
                         liveData = NULL, 
                         stemming = FALSE, 
                         sparsity = NULL,
                         live = FALSE,
                         virgin = FALSE,
                         weight = weightTf) {

        # Read trainingData with columns Class, ID, Text into data frame
        trainingData <- read.table(trainingData, header = TRUE, sep = "\t", quote = "")
        
        # Sort the rows of trainingData randomly in case the
        # model is tested using the first 80% of the rows for the model and
        # the remainder for testing
        input <- trainingData[sample(nrow(trainingData)),]
        
        # Check if there's live data to classify
        if(!is.null(liveData)) {
                # Override default values accordingly
                live = TRUE
                virgin = TRUE

                # Read liveData with columns Class (empty), ID, Text into data frame
                liveData <- read.table(liveData, header = TRUE, sep = "\t", quote = "")
                # Combinde trainingData and liveData into one data frame
                input <- rbind(trainingData[sample(nrow(trainingData)),], liveData)
        }
        
        # Use column Text to classify text
        source <- VectorSource(input$Text)
        
        # Create a text corpus
        corpus <- Corpus(source)
        
        # Clean the created text corpus
        corpus <- tm_map(corpus, content_transformer(tolower)) # Make everyting lovercase
        corpus <- tm_map(corpus, removeNumbers) # Remove numbers
        corpus <- tm_map(corpus, removePunctuation) # Remove punctuation
        corpus <- tm_map(corpus, removeWords, stopwords("english")) # Remove English stopwords
        corpus <- tm_map(corpus, stripWhitespace) # Remove unnecessary whitespace
        if(stemming == TRUE) {
                corpus <- tm_map(corpus, stemDocument, language = "english") # Apply stemming
        }
        
        print("corpus ready...")
        
        # Create document term matrix with the indicated weighting
        dtm <- DocumentTermMatrix(corpus, control = list(weighting = weight))
        
        # Remove sparse words from dtm if a sparsity value is given as an argument
        if(!is.null(sparsity)) {
                dtm <- removeSparseTerms(dtm, sparsity)
        }
        
        print("document term matrix ready...")
        print("creating model...")

        # Get to know your rows!
        totalRows <- nrow(input)
        lastTrainingRow <- numeric()
        firstLiveRow <- numeric()
        
        # If it's about classifying live dat, use all dtm rows coming from trainingData
        # to build the model and classify the text from the remaining rows
        if(live == TRUE) {
                lastTrainingRow <- nrow(trainingData)
                firstLiveRow <- lastTrainingRow + 1
        } else {
                # if it's not about classifying live data, use the first 80% of the dtm
                # to build the model and test it using the remaining 20% 
                lastTrainingRow <- totalRows - round((totalRows * 0.2), digits = 0)
                firstLiveRow <- lastTrainingRow + 1 
        }
        
        # Support Vector Machines
        # -----------------------

        container <- create_container(dtm, input$Class, trainSize = 1:lastTrainingRow, testSize = firstLiveRow:totalRows, virgin = virgin)
        model <- train_model(container, "SVM", kernel = "linear")

        print("model ready")
        
        results <- classify_model(container, model)

        # Write results to data frame
        checkResults <- data.frame(Class = input$Class[firstLiveRow:totalRows], SVM_Class = as.character(results[,"SVM_LABEL"]), SVM_Prob = as.character(results[,"SVM_PROB"]))
        
        # Use ID as row name in the data frame with the results
        row.names(checkResults) <- input$ID[firstLiveRow:totalRows]
        
        # Write results to file
        output = paste("checkResults_", format(Sys.time(), "%Y%m%d%H%M%S"), ".txt", sep = "")
        write.table(as.matrix(checkResults), file = output  , sep = "\t", col.names = NA, quote = FALSE)

        print(paste("results saved as", output))
        
        # Print accuracy of model to consolre if it's not about classifying live data
        if(live == FALSE) {
                accuracy <- recall_accuracy(input$Class[firstLiveRow:totalRows], results[,"SVM_LABEL"])
                print(paste("accuracy:", accuracy))
        }
}
```
