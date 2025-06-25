:::mermaid

graph TD
    A[Start] --> B[Retrieve Features and Activities]
    B --> C[Retrieve ERD Document]
    C --> D[Decide Django Apps & Project Name]
    D --> E[Create Project & Apps under 'generated' Folder]
    E --> F[Create Tooling for File Storage]

    F --> G[Loop through Each Activity]
    G --> G1[Generate Model, View, URL, Form, Template]
    G1 --> G2[Store Files in Corresponding App Folder]
    G2 --> G3[Generate Tests for the Activity]
    G3 --> G4[Run Tests]

    G4 -->|Test Passed| G5[Update Completion Status]
    G4 -->|Test Failed| G6[Log Error or Skip Activity]

    G5 --> H{More Activities?}
    G6 --> H

    H -- Yes --> G
    H -- No --> I[Finish]
