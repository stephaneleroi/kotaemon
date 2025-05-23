# Document Summarization Feature

## Introduction

This feature allows you to automatically generate concise summaries of your documents using advanced language models. You can summarize a single document or an entire directory of documents.

## Accessing the Feature

To access the summarization feature, navigate to the **Summarization** tab in the application.

## Input Fields

When you open the **Summarization** tab, you will see several input fields:

*   **Document Path or Directory**:
    *   Enter the full path to the document you want to summarize.
    *   Alternatively, you can provide a path to a directory. The feature will attempt to summarize all supported files within that directory.
    *   Supported file types include common text-based formats (e.g., .txt, .md, .pdf - *actual supported types may vary*).
*   **LLM for Summarization**:
    *   Choose the language model (LLM) that will be used to generate the summary. Different models may have varying levels of performance and summarization styles.
*   **Target Summary Length (tokens)**:
    *   Use the slider to set the desired length of the generated summary in tokens. A token is roughly equivalent to a word or a part of a word.
*   **Chunk Size (tokens)**:
    *   This input controls the size of text chunks (in tokens) processed at a time. For very large documents, the text is split into smaller chunks to be summarized individually before being combined.
*   **Consolidation Batch Size**:
    *   This setting determines how many individual chunk summaries are combined at once during the consolidation phase.

## Starting Summarization

1.  Once you have configured the input fields, click the **Summarize** button.
2.  Summarization is an asynchronous task, meaning it runs in the background.
    *   You will see status messages such as "Starting summarization..." to indicate that the process has begun.
    *   For large documents or directories with many files, the process might take some time.
    *   A "Summarization complete." message will appear once the task is finished.

## Output

After the summarization process is complete, the following outputs will be displayed:

*   **Summary**:
    *   The generated summary of your document(s) will appear in this section.
*   **Metadata**:
    *   This section provides details about the summarization process, such as the number of chunks processed and the time taken. (This information is primarily for technical reference).

## Troubleshooting/Error Handling

*   If any issues occur during the summarization process (e.g., file not found, unsupported file type), an error message will be displayed to help you identify the problem.
*   Ensure that the file paths are correct and that the files are accessible.
