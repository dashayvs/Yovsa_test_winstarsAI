# Setup Instructions for Both Projects from a Single Repository

projects:
  - name: task_1_ImgCls
    description: Image Classification project.
  - name: task_2_NER_ImgCls
    description: Named Entity Recognition (NER) + Image Classification Pipeline.

steps:
  - step: Clone the Repository
    description: |
      1. Go to the GitHub repository page.
      2. Click on the **"Code"** button and copy the HTTPS or SSH URL.
      3. Run the following command to clone the repository:
         ```bash
         git clone https://github.com/your-username/repository-name.git
         ```
         or, if using SSH:
         ```bash
         git clone git@github.com:your-username/repository-name.git
         ```
      4. Navigate into the cloned repository:
         ```bash
         cd repository-name
         ```

  - step: Create a Virtual Environment
    description: |
      You can use either **Conda** or **venv**.

      ### Using Conda:
      ```bash
      conda create --name ml_pipeline python=3.8
      conda activate ml_pipeline
      ```

      ### Using venv:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Linux/Mac
      # or
      venv\Scripts\activate     # On Windows
      ```

  - step: Install Dependencies
  
      Each project has its own `requirements.txt`, install them separately:
      
      For **task_1_ImgCls**:
      ```bash
      cd task_1_ImgCls
      pip install -r requirements.txt
      ```

      For **task_2_NER_ImgCls**:
      ```bash
      cd ../task_2_NER_ImgCls
      pip install -r requirements.txt
      ```

  - step: Set Up pre-commit Hooks (Optional)
    description: |
      If your repository uses pre-commit hooks (`.pre-commit-config.yaml` file), set them up with these commands:
      1. Install **pre-commit**:
         ```bash
         pip install pre-commit
         ```
      2. Set up the hooks:
         ```bash
         pre-commit install
         ```

  