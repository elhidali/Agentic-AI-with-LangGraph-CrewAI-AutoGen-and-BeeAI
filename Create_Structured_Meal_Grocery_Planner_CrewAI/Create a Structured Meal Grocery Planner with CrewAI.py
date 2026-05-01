
import os
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from crewai import LLM

llm = LLM(
    model=os.getenv('MODEL_ID', 'openrouter/nvidia/nemotron-3-super-120b-a12b:free'),
    base_url=os.getenv('OPENROUTER_BASE_URL'),
    api_key=os.getenv('OPENROUTER_API_KEY'),
    # max_tokens=2000,
)
#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # Create a Structured Meal & Grocery Planner with CrewAI
# 

# Estimated time needed: **45** minutes
# 

# You are an individual who loves cooking delicious meals but struggles with the overwhelming decisions around meal planning, budgeting, and grocery shopping. In the past, this involved hours of manual work – browsing endless recipes, calculating ingredient quantities, checking prices, organizing shopping lists, and hoping everything fits your budget and dietary needs. By leveraging AI agents through CrewAI, you can dramatically accelerate this process while ensuring you get exactly what you need for a perfect meal.
# 
# In this project, you'll implement an end-to-end meal planning and grocery shopping system using AI agents. You'll create:
# 
# - **Structured data models** using Pydantic to ensure consistent grocery lists and meal plans.
# - **Specialized AI agents** for recipe research, shopping organization, and budget optimization.
# - **A coordinated workflow** that transforms a simple meal craving into a complete, budget-conscious shopping strategy.
# - **A reusable YAML configuration** that allows you to define agents, tasks, and crew behavior declaratively, making the system easier to maintain and scale.
# - **A class-based CrewBase setup** that combines structured Python code with external YAML files, enabling hooks for pre-processing inputs and post-processing outputs, and automating the agent-task pipeline.
# 
# By the end of this project, you'll have a reusable framework that can generate comprehensive, organized shopping plans for any meal within minutes rather than hours — all with just a few lines of configuration and code.
# 

# ## __Table of Contents__
# <ol>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
#             <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Creating-the-Grocery-Shopping-Assistant-Structure">Creating the Grocery Shopping Assistant Structure</a>
#         <ol>
#             <li><a href="#GroceryItem">GroceryItem</a></li>
#             <li><a href="#MealPlan">MealPlan</a></li>
#             <li><a href="#ShoppingCategory">ShoppingCategory</a></li>
#             <li><a href="#GroceryShoppingPlan">GroceryShoppingPlan</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Setting-Up-Our-LLM-and-Essential-Tools">Setting Up Our LLM and Essential Tools</a>
#         <ol>
#             <li><a href="#Setting-Up-SerperDevTool">Setting Up SerperDevTool</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Creating-Our-AI-Agent-Workflow-with-CrewAI">Creating Our AI Agent Workflow with CrewAI</a>
#         <ol>
#             <li><a href="#Defining-Our-Meal-Planning-Agent">Defining Our Meal Planning Agent</a></li>
#             <li><a href="#Defining-Our-Meal-Planning-Task">Defining Our Meal Planning Task</a></li>
#             <li><a href="#Creating-and-Running-Our-Meal-Planning-Crew">Creating and Running Our Meal Planning Crew</a></li>
#             <li><a href="#Creating-Our-Shopping-Organization-Agent">Creating Our Shopping Organization Agent</a></li>
#             <li><a href="#Defining-the-Shopping-Organization-Task">Defining the Shopping Organization Task</a></li>
#             <li><a href="#Building-Our-Two-Agent-Grocery-Crew">Building Our Two-Agent Grocery Crew</a></li>
#             <li><a href="#Adding-Financial-Intelligence-with-Budget-Advisor-Agent">Adding Financial Intelligence with Budget Advisor Agent</a></li>
#             <li><a href="#Defining-the-Budget-Analysis-Task">Defining the Budget Analysis Task</a></li>
#             <li><a href="#Using-YAML-with-CrewAI---Food-Leftover-Agent-and-Task">Using YAML with CrewAI - Food Leftover Agent and Task</a></li>
#             <li><a href="#Using-CrewBase-and-Decorators-with-CrewAI">Using CrewBase and Decorators with CrewAI</a></li>
#             <li><a href="#Defining-Our-Summary-Agent-and-Task">Defining Our Summary Agent and Task</a></li>
#             <li><a href="#Assembling-Our-Complete-Grocery-Planning-Team">Assembling Our Complete Grocery Planning Team</a></li>
#             <li><a href="#Executing-Our-Complete-Grocery-Planning-Workflow">Executing Our Complete Grocery Planning Workflow</a></li>
#             <li><a href="#Understanding-Your-Complete-Grocery-Shopping-Guide">Understanding Your Complete Grocery Shopping Guide</a></li>
#         </ol>
#      </li>
#     <li><a href="#Exercises">Exercises</a></li>
#     <li><a href="#Authors">Authors</a></li>
# </ol>
# 

# ## Objectives
# 
# After completing this lab, you will be able to:
# 
# - **Structure grocery planning data** using `Pydantic` models and structured outputs to ensure consistent formats across all agent responses.
# - **Define intelligent food planning agents** using both Python-based `@CrewBase` classes and YAML configuration files for modularity and reuse.
# - **Create crew objects** that assign agents to tasks, enabling coordinated multi-agent workflows for meal planning and grocery management.
# - **Implement a multi-stage planning workflow** with CrewAI that processes user inputs through research, shopping list generation, budgeting, and leftover management.
# - **Organize ingredient data by store sections** to produce structured shopping lists with estimated prices and category-wise grouping.
# - **Compile a complete grocery shopping guide** that combines outputs from all agents into a cohesive report with recipes, cost breakdowns, and money-saving tips.
# 

# ----
# 

# ## Setup
# 

# For this lab, we will be using the following libraries:
# 
# *   [`pydantic`](https://docs.pydantic.dev/latest/) for data validation and creating structured grocery item, meal plan, and shopping list models
# *   [`langchain`](https://python.langchain.com/docs/get_started/introduction) for language model orchestration and integration with recipe research agents
# *   [`crewai`](https://www.crewai.io/) for creating coordinated AI agent workflows that handle meal planning, shopping organization, and budget analysis
# *   [`crewai-tools`](https://github.com/joaomdmoura/crewAI-tools) for specialized tools such as SerperDevTool to enable web-based recipe research and price checking
# *   [`IPython.display`](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html) for rich formatting and display of structured grocery data and shopping plans
# *   [`os`](https://docs.python.org/3/library/os.html) for managing API keys and environment variables needed for web search functionality
# *   [`warnings`](https://docs.python.org/3/library/warnings.html) for suppressing deprecation and user warnings to maintain clean output during meal planning operations
# 

# ### Installing Required Libraries
# 
# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You will need to run the following cell__ to install them:
# 

# In[ ]:


# %pip install langchain==0.3.20 | tail -n 1
# %pip install crewai==0.141.0 | tail -n 1
# %pip install langchain-community==0.3.19 | tail -n 1
# %pip install langchain-openai==0.3.25 | tail -n 1
# %pip install duckduckgo-search==7.5.2 | tail -n 1
# %pip install crewai-tools==0.51.1 | tail -n 1
# %pip install databricks-sdk==0.46.0 | tail -n 1


# In[1]:


import os
import ssl
import urllib3

# Disable SSL verification (required for Zscaler corporate proxy)
_ssl_create_default_context = ssl.create_default_context
def _insecure_create_default_context(*args, **kwargs):
    ctx = _ssl_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _insecure_create_default_context

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

# Fix requests library SSL (used by SerperDevTool and other crewai tools)
import requests.adapters
_original_send = requests.adapters.HTTPAdapter.send
def _insecure_send(self, request, *args, **kwargs):
    kwargs['verify'] = False
    return _original_send(self, request, *args, **kwargs)
requests.adapters.HTTPAdapter.send = _insecure_send

from dotenv import load_dotenv
load_dotenv()


# In[2]:


# !pip install litellm
import litellm
litellm.ssl_verify = False


# We are going to download a python file which contains the `CrewBase` class. The reason why we are downloading it here, will be discussed later.
# 

# In[3]:


# !wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/3xGOgzMOv5jhRsA3A8N9fQ/leftover.py"


# ### Importing Required Libraries
# Since we are creating our own module, Jupyter notebooks might not automatically recognize it — especially in custom or multi-file setups. To ensure our module is importable, we’ll add the current directory to Python’s module search path using sys.path.append('.').
# 

# In[4]:


import sys
sys.path.append(".")


# We'll use ```LeftoversCrew``` later in the lab, but let's double-check it's availability now. If you get an import error, first check that ```leftover.py``` is in your current directory. If it is, try restarting the Jupyter kernel. If the problem persists, double-check that you've written the correct import statement.
# 

# In[5]:


from leftover import LeftoversCrew


# Files in  current directory:
# 

# In[6]:


import os

files = os.listdir('.')
print(files)


# In[7]:


from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from IPython.display import display, JSON, Markdown
from datetime import datetime
import os


# ## Creating the Grocery Shopping Assistant Structure
# 
# To create our grocery shopping assistant, we are going to begin by creating the blueprints (a.k.a. classes) to manage the structure. Why? Because when you're dealing with grocery lists, meal plans, and shopping data, you need everything organized in a predictable format. Without structure, you might get a shopping list that says "some chicken" instead of "2 lbs chicken breast" - not very helpful when you're at the store!
# 
# To do that, we are going to use **Pydantic**, which acts like a strict organizer for our data. Think of it as creating templates that ensure every grocery item has exactly what we need: a name, quantity, price, and store section.
# 
# #### Why Are We Creating These Classes?
# 
# We are creating these classes to **organize and structure grocery shopping data** in a clear and practical way. Instead of handling messy, unstructured text such as "get some chicken and vegetables," we break down shopping into smaller, structured parts that our AI agents can work with reliably.
# 
# Pydantic's `BaseModel` helps structure and validate our grocery data easily. For example, if we define a `GroceryItem` model with `name: str` and `estimated_price: str`, it ensures that every grocery item must have both a name and a price estimate. `Field` adds extra rules and descriptions, like explaining that `quantity` should be formatted as "2 lbs" or "1 gallon". `List` from `typing` allows complex data structures, like a `ShoppingCategory` model storing `items: List[GroceryItem]` - a produce section containing multiple vegetables.
# 
# This makes handling grocery data simple and reliable, ensuring that when our AI agent says "buy chicken," it specifically means "buy 2 lbs of chicken breast for $8-12 from the meat section." ([Pydantic Docs](https://docs.pydantic.dev/latest/concepts/models/))
# 

# ### Our Grocery Shopping Data Structure
# 
# 1. ```GroceryItem```         → Individual grocery item with details
# 2. ```MealPlan```           → Recipe information with researched ingredients  
# 3. ```ShoppingCategory```   → Store section with organized items
# 4. ```GroceryShoppingPlan``` → Complete shopping strategy with budget analysis
# 

# ### **`GroceryItem`** 
# 
# Think of this as designing the shopping list template before going to the store. BaseModel gives us the foundation, Field decorates our items with clear descriptions. This structure helps both humans and AI understand grocery data consistently, and includes several key string inputs:
# 
# - **`name`** – The specific grocery item to purchase (for example, "Chicken Breast").
# - **`quantity`** – How much to buy in clear measurements (for example, "2 lbs", "1 gallon").
# - **`estimated_price`** – The expected cost range for budgeting (for example, "$8-12").
# - **`category`** – Which store section to find it in (for example, "Meat", "Produce", "Dairy").
# 

# In[8]:


class GroceryItem(BaseModel):
    """Individual grocery item"""
    name: str = Field(description="Name of the grocery item")
    quantity: str = Field(description="Quantity needed (for example, '2 lbs', '1 gallon')")
    estimated_price: str = Field(description="Estimated price (for example, '$3-5')")
    category: str = Field(description="Store section (for example, 'Produce', 'Dairy')")


# We create an instance of a `GroceryItem`, which holds **one specific grocery item** along with its purchase details and store location.  
# 
# This **standard format** helps our AI shopping organizer stay organized and maintain consistency when creating shopping lists.
# 

# In[9]:


sample_item = GroceryItem(
    name="Chicken Breast",
    quantity="2 lbs",
    estimated_price="$8-12",
    category="Meat"
)


# The result is essentially a **JSON object** or a **Python dictionary**.  
# 
# Without formatting, our `sample_item` displays as:
# 

# In[10]:


sample_item


# In[11]:


type(sample_item)


# We can use the `display` function to **view the output in a much cleaner format**, especially noting its **organized, readable structure**. Since this is a single grocery item, all fields become clearly visible—**each property shows exactly what information our AI agents will work with**.
# 

# In[12]:


# Display structured data
print("🛒 Sample Grocery Item Structure:")
display(JSON(sample_item.model_dump()))


# This class lets us **create multiple grocery items**. Here, we're generating a **sample grocery item for chicken breast** so we can demonstrate how individual items are structured.
# 
# You can create multiple grocery items to build complete shopping lists:
# - The first might cover proteins - essential ingredients for main dishes
# - The second could focus on vegetables that provide nutrition and flavor  
# - A third might include pantry staples such as rice or cooking oils
# 

# ### **`MealPlan`** 
# 
# It represents a complete meal with all the details needed for cooking, including:
# 
# - **`meal_name`** – The name of the dish being prepared (for example, "Chicken Stir Fry").
# - **`difficulty_level`** – How challenging it is to cook ("Easy", "Medium", "Hard").
# - **`servings`** – Number of people the meal will feed (integer value).
# - **`researched_ingredients`** – List of ingredients found through AI research (note this is a list).
# 

# In[13]:


class MealPlan(BaseModel):
    """Simple meal plan"""
    meal_name: str = Field(description="Name of the meal")
    difficulty_level: str = Field(description="'Easy', 'Medium', 'Hard'")
    servings: int = Field(description="Number of people it serves")
    researched_ingredients: List[str] = Field(description="Ingredients found through research")


# Similarly, we can create an instance of a `MealPlan`, which holds **one complete meal** along with its cooking difficulty and researched ingredient list.  
# 

# In[14]:


sample_meal = MealPlan(
    meal_name="Chicken Stir Fry",
    difficulty_level="Easy",
    servings=4,
    researched_ingredients=["chicken breast", "broccoli", "bell peppers", "garlic", "soy sauce", "rice"]
)


# In[15]:


sample_meal


# In[16]:


from pprint import pprint

pprint("\n🍽️ Sample Meal Plan Structure:")
display(JSON(sample_meal.model_dump()))


# ### **`ShoppingCategory`** 
# 
# It organizes items by store layout for efficient shopping, containing:
# 
# - **`section_name`** – The store department name (for example, "Produce", "Meat & Poultry").
# - **`items`** – A collection of GroceryItem objects in this section (note this is a list).
# - **`estimated_total`** – The expected cost for all items in this category.
# 

# In[17]:


class ShoppingCategory(BaseModel):
    """Store section with items"""
    section_name: str = Field(description="Store section (for example, 'Produce', 'Dairy')")
    items: List[GroceryItem] = Field(description="Items in this section")
    estimated_total: str = Field(description="Estimated cost for this section")


# Below, we create an instance of a `ShoppingCategory`, which holds **one store section** along with multiple grocery items and a cost estimate.  
# Notice how `items: List[GroceryItem]` allows us to **group multiple grocery items together** - this is where our individual `GroceryItem` objects get organized by store layout. Instead of a single item, we can include an entire list of items that belong in the same store section, making shopping more efficient.
# 

# In[18]:


sample_section = ShoppingCategory(
    section_name="Produce",
    items=[
        GroceryItem(name="Bell Peppers", quantity="3 pieces", estimated_price="$3-4", category="Produce"),
        GroceryItem(name="Onions", quantity="2 lbs", estimated_price="$2-3", category="Produce")
    ],
    estimated_total="$5-7"
)


# In[19]:


pprint("\n🏪 Sample Shopping Section:")
display(JSON(sample_section.model_dump()))


# ### **`GroceryShoppingPlan`** 
# 
# It is the master plan that combines everything into a complete shopping strategy:
# 
# - **`total_budget`** – The overall spending limit for the shopping trip.
# - **`meal_plans`** – Collection of meals being prepared (note this is a list).
# - **`shopping_sections`** – Organized categories for store navigation (note this is a list).
# - **`shopping_tips`** – Money-saving advice and practical suggestions (note this is a list).
# 

# In[20]:


class GroceryShoppingPlan(BaseModel):
    """Complete simplified shopping plan"""
    total_budget: str = Field(description="Total planned budget")
    meal_plans: List[MealPlan] = Field(description="Planned meals")
    shopping_sections: List[ShoppingCategory] = Field(description="Organized by store sections")
    shopping_tips: List[str] = Field(description="Money-saving and efficiency tips")


# We can create a `GroceryShoppingPlan` object—this represents the **complete shopping strategy**, compiled from the meal plans and shopping sections we've created. Below is the just an example of how it looks and **not an executable code**.
# 
# weekly_shopping_plan = GroceryShoppingPlan(
#     total_budget="$40-50",
#     meal_plans=[breakfast_meal, lunch_meal, dinner_meal],  # List of MealPlan objects
#     shopping_sections=[dairy_section, meat_section, pantry_section, produce_section],  # List of ShoppingCategory objects
#     shopping_tips=[...]
# )
# # ## Setting Up Our LLM and Essential Tools  
# # 
# We set up our **LLM (Large Language Model)**—this can be **any model** based on our needs. Here, we are going to use **Granite 3.3**. 
# 
# > This lab uses LLM provided by **IBM WatsonX**. This environment has been configured to allow LLM use without API keys so you can prompt them for **free (with limitations)**. With that in mind, if you wish to run this notebook **locally outside** of Skills Network's JupyterLab environment, you will have to **configure your own API keys**. Please note that using your own API keys means that you will incur personal charges. 
# 

# In[37]:




# ### Setting Up SerperDevTool  
# 
# **What is Serper?**  
# Serper is a Google Search API that provides real-time web search capabilities to AI applications. It acts as a bridge between our AI agents and the vast information available on the internet.
# 
# **Why are we using Serper in our workflow?**  
# Our meal planning agents need access to current recipe information, ingredient prices, cooking techniques, and dietary alternatives. Without web access, our AI would be limited to outdated training data and couldn't find the latest recipes or seasonal ingredient information that makes meal planning practical and relevant.
# 
# **How will we use Serper?**  
# Our `meal_planner` agent will use Serper to:
# - Search for recipes that match specific dietary restrictions (for example, "gluten-free chicken stir fry").
# - Find cooking techniques appropriate for different skill levels (for example, "beginner-friendly stir fry methods").
# - Research ingredient substitutions and alternatives.
# - Check for seasonal availability and pricing information.
# 
# To use `SerperDevTool`, we first need an **API key**—a crucial credential that grants access to web searches. Once we have the API key, we can import the library and integrate it into our meal planning workflow.  
# 
# > You will need to get the API Key from `serper.dev` website. To get the API key, create an account using your email if you have registered or login if you already have an account. Once you are logged in, you will be on the `Dashboard` section. Click on `API Keys` and you will find your API key. Copy the key and replace it with `API_KEY` here.
# 
# To get the API key and for more details on `SerperDevTool` and its capabilities, visit the official webpage: [SerperDevTool Documentation](https://serper.dev/).
# 

# In[22]:


# Set up search tool (you'll need to add your API key)
# os.environ['SERPER_API_KEY'] = 'API_KEY'  # Replace with actual key


# #### Meal and Grocery Planning Workflow Overview
# 
# ![image (3).png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/0crjoLc3Yn0JefQoJVUarA/image%20-3-.png)
# 
# ![image.png](attachment:image.png)
# 

# ## Creating Our AI Agent Workflow with CrewAI
# 
# We are going to use **CrewAI** to create our intelligent meal planning workflow. CrewAI allows us to build teams of specialized AI agents that work together seamlessly. We'll start by creating our first agent.
# 
# ### Defining Our Meal Planning Agent  
# We are creating our first **AI meal planning specialist**, responsible for researching recipes and creating detailed meal plans.  
# - **Role and Goal** – The agent is assigned a clear identity as a "Meal Planner & Recipe Researcher" with the specific purpose of finding optimal recipes, ensuring focused culinary research.  
# - **Backstory** – This provides context for how the agent approaches meal planning tasks, much like defining expertise in a professional chef or nutritionist who considers dietary needs and skill levels.  
# - **Tools** – The agent is equipped with web search capabilities through `SerperDevTool()`, enabling access to real-time recipe information, cooking tips, and ingredient data.  
# - **Verbose Mode (`verbose=False`)** – Stops detailed logs of the agent's thought process, keeping output clean and focused.
# 

# In[38]:


meal_planner = Agent(
    role="Meal Planner & Recipe Researcher",
    goal="Search for optimal recipes and create detailed meal plans",
    backstory="A skilled meal planner who researches the best recipes online, considering dietary needs, cooking skill levels, and budget constraints.",
    tools=[SerperDevTool()],
    llm=llm,
    verbose=False
)


# ### Defining Our Meal Planning Task
# 
# **What is a Task in CrewAI?**  
# A Task in CrewAI is a specific assignment given to an AI agent. It defines what the agent should do, what inputs it will receive, and what output is expected. Tasks are the building blocks that turn AI agents into productive team members with clear responsibilities.
# 
# We are assigning a structured meal planning task to our AI agent, ensuring clear expectations and a standardized output format. The `{meal_name}`, `{servings}`, `{budget}`, `{dietary_restrictions}`, and `{cooking_skill}` placeholders make the task dynamic and reusable, allowing it to adapt to different meal requests and personal preferences. Additionally, the results are getting saved in "meals.json" which is optional.
# 

# In[39]:


meal_planning_task = Task(
    description=(
        "Search for the best '{meal_name}' recipe for {servings} people within a {budget} budget. "
        "Consider dietary restrictions: {dietary_restrictions} and cooking skill level: {cooking_skill}. "
        "Find recipes that match the skill level and provide complete ingredient lists with quantities."
    ),
    expected_output="A detailed meal plan with researched ingredients, quantities, and cooking instructions appropriate for the skill level.",
    agent=meal_planner,
    output_pydantic=MealPlan,
    output_file="meals.json"
)


# ### Creating and Running Our Meal Planning Crew
# 
# **What is a Crew in CrewAI?**  
# A Crew is like assembling a team of specialists to work on a project. Just as you might have a chef, a shopping assistant, and a budget manager working together to plan a meal, a CrewAI Crew coordinates multiple AI agents, each with their own tasks, to accomplish a complex workflow. The Crew manages how agents collaborate, share information, and execute their tasks in the right order.
# 
# We initialize a CrewAI workflow with a single meal planning agent, executing the recipe research task in a sequential process, where the AI gathers structured meal insights for the specified recipe and requirements before storing the results. However, this is only a partial output, as we will later integrate all the Pydantic `BaseModel` objects defined earlier to structure the complete shopping plan.
# 

# In[40]:


from crewai import Crew, Process

meal_planner_crew = Crew(
    agents=[meal_planner],
    tasks=[meal_planning_task],
    process=Process.sequential,  # Ensures tasks are executed in order
    verbose=True
)

meal_planner_result = meal_planner_crew.kickoff(
    inputs={
        "meal_name": "Chicken Stir Fry",
        "servings": 4,
        "budget": "$25",                           
        "dietary_restrictions": ["no nuts"],       
        "cooking_skill": "beginner"                
    }
)
print("✅ Single meal planning completed!")
print("📋 Single Meal Results:")
print(meal_planner_result)


# ### Creating Our Shopping Organization Agent
# 
# Now that we have a meal planner that can research recipes, we need a second specialist to transform those meal plans into organized shopping lists. This is where our **Shopping Organizer** agent comes in.
# 
# **Why do we need a separate shopping organizer?**
# - **Specialization**: While the meal planner focuses on recipes and cooking, the shopping organizer specializes in store logistics and efficient shopping
# - **Store Knowledge**: This agent understands how grocery stores are organized (produce, dairy, meat sections) and can group items accordingly
# - **Quantity Management**: It can calculate proper quantities for different serving sizes and ensure nothing is forgotten
# 
# **Key Features of Our Shopping Organizer:**
# - **No External Tools**: Unlike the meal planner, this agent doesn't need web search - it works with the meal plan data internally
# - **Store Section Expertise**: Groups items by where you'll find them in the store (produce, dairy, meat, pantry)
# - **Quantity Calculation**: Ensures proper amounts for the specified number of servings
# - **Budget Awareness**: Considers the overall budget when organizing the shopping list
# 

# In[41]:


shopping_organizer = Agent(
    role="Shopping Organizer", 
    goal="Organize grocery lists by store sections efficiently",
    backstory="An experienced shopper who knows how to organize lists for quick store trips and considers dietary restrictions.",
    tools=[],
    llm=llm,
    verbose=False
)


# ### Defining the Shopping Organization Task
# 
# The shopping task is where we connect our meal planning output to our shopping organization needs. This task takes the researched meal plan and transforms it into a practical shopping list.
# 
# **Key Elements of the Shopping Task:**
# - **Context Dependency**: Uses `context=[meal_planning_task]` to access the meal planner's output
# - **Pydantic Output**: Uses `output_pydantic=GroceryShoppingPlan` to ensure structured, validated output
# - **Store Organization**: Groups ingredients by store sections for efficient shopping
# - **Budget Integration**: Keeps track of costs and stays within the specified budget
# 
# **The Power of Context in CrewAI:**
# When we set `context=[meal_planning_task]`, the shopping organizer can access all the research and ingredients found by the meal planner. This creates a seamless workflow where each agent builds upon the previous agent's work.
# 

# In[42]:


shopping_task = Task(
    description=(
        "Organize the ingredients from the '{meal_name}' meal plan into a grocery shopping list. "
        "Group items by store sections and estimate quantities for {servings} people. "
        "Consider dietary restrictions: {dietary_restrictions} and cooking skill: {cooking_skill}. "
        "Stay within budget: {budget}."
    ),
    expected_output="An organized shopping list grouped by store sections with quantities and prices.",
    agent=shopping_organizer,
    context=[meal_planning_task],
    output_pydantic=GroceryShoppingPlan,
    output_file="shopping_list.json"
)


# ### Building Our Two-Agent Grocery Crew
# 
# Now we're ready to create our first multi-agent workflow. This crew combines both the meal planner and shopping organizer to create a complete meal-to-shopping-list pipeline.
# 
# **How Multi-Agent Workflows Work:**
# 1. **Sequential Processing**: The `Process.sequential` ensures agents work in order - meal planning first, then shopping organization
# 2. **Data Flow**: The shopping organizer automatically receives the meal planner's output through the task context
# 3. **Structured Output**: The final result follows our `GroceryShoppingPlan` Pydantic model for consistent formatting
# 
# **Benefits of This Approach:**
# - **Separation of Concerns**: Each agent has a specific role and expertise
# - **Reusability**: We can swap out agents or add new ones without changing the overall workflow
# - **Reliability**: Pydantic ensures our output is always properly structured
# 

# In[43]:


two_agent_grocery_crew = Crew(
    agents=[meal_planner, shopping_organizer],  # Both agents
    tasks=[meal_planning_task, shopping_task],   # Both tasks
    process=Process.sequential,
    verbose=True
)

# Run the complete crew (this will do BOTH meal planning AND shopping)
shopping_result = two_agent_grocery_crew.kickoff(
    inputs={
        "meal_name": "Chicken Stir Fry",
        "servings": 4,
        "budget": "$25",                           
        "dietary_restrictions": ["no nuts"],      
        "cooking_skill": "beginner"               
    }
)

# Print the shopping results
print("✅ Complete meal planning + shopping completed!")
print("📋 Shopping Results:")
print(shopping_result)


# ### Adding Financial Intelligence with Budget Advisor Agent
# 
# Our meal planning and shopping organization is great, but we need someone to watch the budget and provide money-saving advice. Enter our **Budget Advisor** - the financial expert of our grocery team.
# 
# **Why Add a Budget Advisor?**
# - **Cost Control**: Ensures the shopping plan stays within budget limits
# - **Price Research**: Uses web search to find current prices and deals
# - **Smart Alternatives**: Suggests cheaper ingredient substitutions when needed
# - **Money-Saving Tips**: Provides practical advice for reducing grocery costs
# 
# **Budget Advisor's Unique Features:**
# - **Web Search Access**: Uses `SerperDevTool()` to research current prices and deals
# - **Context Awareness**: Has access to both meal planning and shopping organization results
# - **Practical Focus**: Provides actionable advice rather than just price estimates
# 

# In[44]:


budget_advisor = Agent(
    role="Budget Advisor",
    goal="Provide cost estimates and money-saving tips",
    backstory="A budget-conscious shopper who helps families save money on groceries while respecting dietary needs.",
    tools=[SerperDevTool()],
    llm=llm,
    verbose=False
)


# ### Defining the Budget Analysis Task
# 
# The budget task is our financial checkpoint - it takes the complete shopping plan and adds cost analysis, budget verification, and money-saving recommendations.
# 
# **What Makes This Task Special:**
# - **Multiple Context Sources**: Uses `context=[meal_planning_task, shopping_task]` to access both previous outputs
# - **Financial Analysis**: Calculates total costs and compares against budget limits
# - **Alternative Suggestions**: Provides cheaper ingredient options when budget is tight
# - **Practical Tips**: Offers real-world money-saving strategies
# 
# **The Context Chain:**
# By now, our budget advisor has access to:
# 1. Original meal research from the meal planner
# 2. Organized shopping lists from the shopping organizer
# 3. Current price data from web searches
# 
# This creates a comprehensive understanding of the entire grocery shopping scenario.
# 

# In[45]:


budget_task = Task(
    description=(
        "Analyze the shopping plan for '{meal_name}' serving {servings} people. "
        "Ensure total cost stays within {budget}. Consider dietary restrictions: {dietary_restrictions}. "
        "Provide practical money-saving tips and alternative ingredients if needed to meet budget."
    ),
    expected_output="A complete shopping guide with detailed prices, budget analysis, and money-saving tips.",
    agent=budget_advisor,
    context=[meal_planning_task, shopping_task],
    output_file="shopping_guide.md"
)


# ###  Using YAML with CrewAI - Food Leftover Agent and Task
# 
# ####  What is YAML?
# 
# **YAML** (YAML Ain’t Markup Language) is a lightweight, human-readable format for structuring data. It is widely used in configuration files because it:
# 
# - Is easy to read and write.
# - Supports nested data structures (for example, lists, dictionaries).
# - Keeps logic separate from code, ideal for configuration.
# 
# ####  Why Use YAML with CrewAI?
# 
# CrewAI supports YAML-based configuration for defining:
# - Agents
# - Tasks
# - Crew processes
# 
# This allows you to:
# - Quickly iterate on agent/task logic without modifying Python code.
# - Enable team collaboration (even with non-developers).
# - Prepare workflows for production deployment.
# 
# #### What Can You Define in YAML?
# 
# In YAML you can define te following:
# 
# - **Agents**: Roles, goals, backstories, tools, verbosity, and more
# - **Tasks**: Task descriptions, expected outputs, assigned agents, and dependencies
# - **Crew**: The execution flow (for example, `sequential`, `hierarchical`)
# 
# #### YAML Example: Meal Planning Crew
# 
# #### `agents.yaml`
# ![image (1).png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/K-z4IjZnpV2pnhiAGpSvmw/image%20-1-.png)
# 
# ![image.png](attachment:image.png)
# 
# #### `tasks.yaml`
# ![image (2).png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/HJ7axigzuQu8tdRR4UQMCg/image%20-2-.png)
# 
# ![image-2.png](attachment:image-2.png)
# 

# Here, we have already created a YAML file for food leftover agent and task for your convenience. You can download it using below code.
# 

# In[46]:


# !wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/-5NXnHGkX1WG81tqE8kSTg/agents.yaml"


# In[47]:


# !wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7fhq1KPxi7cuqjOCBgcmXQ/tasks.yaml"


# > On the left of your screen you can notice the file manager. In that we have the downloaded .yaml files. Double-click on them to open it in the new window to see how the agent and task are configured. 
# 

# Let's move the downloaded `agent.yaml` and `tasks.yaml` under `config` folder.
# 

# In[48]:


# %mkdir -p config
# %mv agents.yaml  config/agents.yaml
# %mv tasks.yaml   config/tasks.yaml


# > Note, for convenience we already created a separate `leftover.py` which has the `CrewBase` class which we already downloaded at the beginning of the lab.
# 

# ### Using CrewBase and Decorators with CrewAI
# 
# #### What is `@CrewBase`?
# In CrewAI, `@CrewBase` is a Python class decorator that automates the collection and wiring of your agents and tasks — especially when you're organizing your crew logic in Python, YAML or hybrid config (YAML + code). Basically, when we have the following:
# 
# ```python
# from crewai.project import CrewBase
# from crewai.project.annotations import agent, task
# 
# @CrewBase
# class myCrew:
#     ...
# ```
# It tells CrewAI that: This class defines the agents, tasks, and crew I want to run. Please find them automatically and wire everything together.
# 
# You might think that we already defined separate Agent, Task and Crew why do we need CrewBase? We need it because this can get repetitive and hard to manage as your app grows. That’s where `@CrewBase` shines.
# 
# ---
# 
# #### Why use `@CrewBase` + decorators?
# - **Auto-discovers** methods decorated with @agent, @task, and @crew inside the class
# - **Loads config** automatically from YAML (if present)
# - Keeps everything **modular**, **scalable**, and **clean** for production setups
# 
# ---
# 
# #### Decorators overview
# - CrewAI provides several decorators that are used to mark methods within your crew class for special handling.
# | Decorator           | Marks…                                            |
# | :------------------ | :----------------------------------------------- |
# | `@CrewBase`         | The class that holds your agents & tasks         |
# | `@agent`            | A method that returns an `Agent` object          |
# | `@task`             | A method that returns a `Task` object            |
# | `@crew`             | (Optional) A method that returns a `Crew` object |
# | `@before_kickoff`   | (Optional) Runs once **before** the crew starts  |
# | `@after_kickoff`    | (Optional) Runs once **after** the crew finishes |
# 
# > The **CrewBase class**, along with these decorators, automates the collection of agents and tasks, reducing the need for manual management.
# 

# ### How `@CrewBase` Works with YAML in This Project
# 
# CrewAI's `@CrewBase` decorator works seamlessly with YAML files, pure Python, or a hybrid setup — and in our case, we're using the hybrid approach to demonstrate how it all comes together.
# 
# So, what's happening here is:
# - The `@agent` method is loading the agent config from the YAML file via: `self.agents_config["leftover_manager"]` and returns an Agent object.
# - Similarly, the `@task` method does: `self.tasks_config["leftover_task"]` and returns a Task object, linked to the `leftover_manager`.
# 
# In our Jupyter notebook, we import the `LeftoverCrew` class which allows us to access the Agent and Task objects that were dynamically constructed using YAML-based config.
# 
# ---
# 
# #### Why YAML Location Matters?
# The `@CrewBase` class (under the hood) looks for default YAML files under a `config/` directory. That’s why we needed to place our `agents.yaml` and `tasks.yaml` inside a `config/` folder — otherwise CrewAI wouldn’t find them automatically.
# 
# ---
# 
# #### Why We Needed a .py File?
# The @CrewBase class and its decorators such as `@agent`, `@task`, and `@crew` do not work in Jupyter notebooks due to how the CrewAI framework relies on Python’s inspection tools. What it means is CrewAI uses Python’s `inspect` module to find the file path of the class:
# ```python
# Path(inspect.getfile(cls)).parent
# ```
# This works fine when the class is defined in a regular `.py` file.
# 
# But in Jupyter notebooks:
# - Classes are defined in cells, not .py files.
# - `inspect.getfile(cls)` fails because there is no real source file.
# - As a result, CrewAI cannot determine the base directory, and throws an error.
# 
# To fix this, we moved the crew class into a separate `.py` file (leftover.py).
# 

# Here is how the `@CrewBase` class from `leftover.py` looks:
# 
# ![image.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/36nCxdNoEA3stoy9uM5WaQ/image.png)
# 
# ![image.png](attachment:image.png)
# 

# 

# 

# In[49]:


from leftover import LeftoversCrew

leftovers_cb = LeftoversCrew(llm=llm)
yaml_leftover_manager = leftovers_cb.leftover_manager()
yaml_leftover_task    = leftovers_cb.leftover_task()


# > 💡 **Note:** While `@CrewBase` makes your agent/task management easier, it wraps your class behind the scenes.
# >
# > That means:
# > - You can't always use `__init__()` or pass custom objects like in normal Python classes.
# > - Testing and debugging `@CrewBase` classes inside Jupyter notebooks can be tricky due to how Jupyter handles decorators and dynamic reloading.
# >
# > 📁 For stability, define your `@CrewBase` classes in separate `.py` files instead of notebooks. This makes CrewAI's configuration loading and agent resolution work reliably.
# >
# > 🔍 See the [CrewAI documentation](https://docs.crewai.com/) for full details on how `@CrewBase` works.
# 

# ### Defining Our Summary Agent and Task
# 
# To manage all the content generated by the agents, we will now create one final agent and its corresponding task which will gather all the content and create a detailed summary. 
# 

# In[50]:


summary_agent = Agent(
    role="Report Compiler",
    goal="Compile comprehensive meal planning reports from all team outputs",
    backstory="A skilled coordinator who organizes information from multiple specialists into comprehensive, easy-to-follow reports.",
    tools=[],
    llm=llm,
    verbose=False
)


# In[51]:


summary_task = Task(
    description=(
        "Compile a comprehensive meal planning report that includes:\n"
        "1. The complete recipe and cooking instructions from the meal planner\n"
        "2. The organized shopping list with prices from the shopping organizer\n"
        "3. The budget analysis and money-saving tips from the budget advisor\n"
        "4. The leftover management suggestions from the waste reduction specialist\n"
        "Format this as a complete, user-friendly meal planning guide."
    ),
    expected_output="A comprehensive meal planning guide that combines all team outputs into one cohesive report.",
    agent=summary_agent,
    context=[meal_planning_task, shopping_task, budget_task, yaml_leftover_task],
)


# ### Assembling Our Complete Grocery Planning Team
# 
# Next, we bring together all five specialists into one powerful grocery planning crew. This represents our complete end-to-end solution from meal idea to comprehensive shopping guide with zero-waste strategies.
# 
# **Our Five-Agent Team:**
# 1. **Meal Planner**: Researches recipes and creates meal plans
# 2. **Shopping Organizer**: Transforms meal plans into organized shopping lists  
# 3. **Budget Advisor**: Adds financial analysis and money-saving strategies
# 4. **Leftover Manager**: Minimizes food waste by suggesting creative uses for leftover ingredients
# 5. **Report Compiler**: Consolidates all outputs into one comprehensive, user-friendly guide
# 

# In[52]:


complete_grocery_crew = Crew(
    agents=[
        meal_planner,           
        shopping_organizer,      
        budget_advisor,         
        yaml_leftover_manager,  # YAML-based leftover manager
        summary_agent           # New summary agent
    ],
    tasks=[
        meal_planning_task,     
        shopping_task,          
        budget_task,            
        yaml_leftover_task,    # YAML-based leftover task
        summary_task            # New summary task
    ],
    process=Process.sequential,
    verbose=True
)


# ### Executing Our Complete Grocery Planning Workflow
# 
# It's now time to put our five-agent team to work! We're going to run the complete workflow with more detailed inputs to demonstrate the full capabilities of our system.
# 
# **Enhanced Input Parameters:**
# - **Multiple Dietary Restrictions**: `["no nuts", "low sodium"]` shows how the system handles complex dietary needs
# - **Skill Level Consideration**: `"beginner"` ensures recipe complexity matches cooking ability
# - **Budget Constraints**: `"$25"` provides a realistic budget limit for practical planning
# - **Serving Size**: `4 people` for family meal planning
# - **Meal Focus**:`Chicken Stir Fry` as our target dish
# 
# **What Happens During Execution:**
# 1. **Meal Planner** searches for beginner-friendly, nut-free, low-sodium chicken stir fry recipes
# 2. **Shopping Organizer** creates a structured shopping list organized by store sections
# 3. **Budget Advisor** analyzes costs, ensures budget compliance, and provides money-saving tips
# 4. **Leftover Manager** identifies ingredients that might result in leftovers and suggests creative bonus recipes to minimize waste and maximize the grocery budget
# 5. **Report Compiler** consolidates all outputs into one comprehensive, user-friendly guide with clear sections for easy reference
# 
# The result is a comprehensive grocery shopping guide that considers all your constraints and preferences.
# 

# In[54]:


# Run the complete crew
complete_result = complete_grocery_crew.kickoff(
    inputs={
        "meal_name": "Chicken Stir Fry",
        "servings": 4,
        "budget": "$25",
        "dietary_restrictions": ["no nuts", "low sodium"],
        "cooking_skill": "beginner"
    }
)

print("✅ Complete meal planning with summary completed!")
print("📋 Complete Results:")
print(complete_result)


# ### Understanding Your Complete Grocery Shopping Guide
# 
# Congratulations! You've successfully created an AI-powered grocery shopping assistant that transforms a simple meal request into a comprehensive shopping strategy.
# 
# **What You've Accomplished:**
# - **Automated Recipe Research**: No more browsing endless recipes online
# - **Organized Shopping Lists**: Items grouped by store sections for efficient shopping
# - **Budget Management**: Cost analysis and money-saving recommendations
# - **Dietary Compliance**: All suggestions respect your dietary restrictions and cooking skill level
# - **Waste Reduction**: Creative leftover management with bonus recipes to maximize your grocery budget
# - **Comprehensive Reporting**: All information compiled into one easy-to-follow guide
# 
# **Next Steps:**
# You can now adapt this framework for any cuisine, dietary need, or budget constraint. Try different meal requests, adjust serving sizes, experiment with various dietary restrictions, or test different cooking skill levels to see how the system adapts. The five-agent team will handle everything from recipe research to waste reduction, ensuring you get maximum value from every grocery trip.
# 

# # Exercises
# 
# It's now time to apply what you've learned! These exercises will help you extend the grocery planning system and understand how to adapt it for different scenarios.
# 

# ### Exercise 1 - Create a Specialized Dietary Agent
# 
# **Objective:** 
# 
# Add a fourth agent that specializes in dietary analysis and nutritional recommendations.
# 
# **Create a new agent called nutrition_analyst that:**
# * Analyzes the nutritional content of meal plans.
# * Suggests healthier alternatives when needed.
# * Provides calorie and macronutrient estimates.
# * Considers specific dietary goals (weight loss, muscle building, etc).
# 
# **Requirements:**
# 1. Create the agent with appropriate role, goal, and backstory
# 2. Define a task that analyzes the meal plan for nutritional value
# 3. Add it to your crew workflow
# 4. Test it with a meal request that includes nutritional goals
# 

# In[ ]:


# # Your code here
# nutrition_analyst = Agent(
#     role="",  # TODO: Define the role
#     goal="",  # TODO: Define the goal  
#     backstory="",  # TODO: Create backstory
#     tools=[],  # TODO: Add appropriate tools
#     llm=llm,
#     verbose=False
# )

# # TODO: Create nutrition analysis task
# # TODO: Add to crew and test


nutrition_analyst = Agent(
    role="Nutrition Analyst & Health Advisor",
    goal="Analyze meal nutritional content and provide healthy recommendations",
    backstory="A certified nutritionist who evaluates meals for nutritional balance, calorie content, and health optimization while respecting dietary restrictions.",
    tools=[SerperDevTool()],
    llm=llm,
    verbose=False
)

nutrition_task = Task(
    description=(
        "Analyze the nutritional content of the '{meal_name}' meal plan for {servings} people. "
        "Calculate approximate calories, protein, carbs, and fats. Consider dietary restrictions: {dietary_restrictions}. "
        "Provide healthy alternatives if the meal could be more nutritious while staying within {budget}."
    ),
    expected_output="Nutritional analysis with calorie estimates, macronutrient breakdown, and healthy improvement suggestions.",
    agent=nutrition_analyst,
    context=[meal_planning_task, shopping_task, budget_task],
    output_file="nutrition_analysis.md"
)

# Test with expanded crew
health_focused_crew = Crew(
    agents=[meal_planner, shopping_organizer, budget_advisor, nutrition_analyst, yaml_leftover_manager, summary_agent],
    tasks=[meal_planning_task, shopping_task, budget_task, nutrition_task, yaml_leftover_task, summary_task],
    process=Process.sequential,
    verbose=True
)

result = health_focused_crew.kickoff(
    inputs={
        "meal_name": "Quinoa Buddha Bowl",
        "servings": 2,
        "budget": "$20",
        "dietary_restrictions": ["vegetarian", "high protein"],
        "cooking_skill": "intermediate"
    }
)


# <details>
#     <summary>Click here for the solution</summary>
# 
# ```python
# nutrition_analyst = Agent(
#     role="Nutrition Analyst & Health Advisor",
#     goal="Analyze meal nutritional content and provide healthy recommendations",
#     backstory="A certified nutritionist who evaluates meals for nutritional balance, calorie content, and health optimization while respecting dietary restrictions.",
#     tools=[SerperDevTool()],
#     llm=llm,
#     verbose=False
# )
# 
# nutrition_task = Task(
#     description=(
#         "Analyze the nutritional content of the '{meal_name}' meal plan for {servings} people. "
#         "Calculate approximate calories, protein, carbs, and fats. Consider dietary restrictions: {dietary_restrictions}. "
#         "Provide healthy alternatives if the meal could be more nutritious while staying within {budget}."
#     ),
#     expected_output="Nutritional analysis with calorie estimates, macronutrient breakdown, and healthy improvement suggestions.",
#     agent=nutrition_analyst,
#     context=[meal_planning_task, shopping_task, budget_task],
#     output_file="nutrition_analysis.md"
# )
# 
# # Test with expanded crew
# health_focused_crew = Crew(
#     agents=[meal_planner, shopping_organizer, budget_advisor, nutrition_analyst, yaml_leftover_manager, summary_agent],
#     tasks=[meal_planning_task, shopping_task, budget_task, nutrition_task, yaml_leftover_task, summary_task],
#     process=Process.sequential,
#     verbose=True
# )
# 
# result = health_focused_crew.kickoff(
#     inputs={
#         "meal_name": "Quinoa Buddha Bowl",
#         "servings": 2,
#         "budget": "$20",
#         "dietary_restrictions": ["vegetarian", "high protein"],
#         "cooking_skill": "intermediate"
#     }
# )
# ```
# </details>
# 

# ### Exercise 2 - Extend Pydantic Models for Weekly Planning
# 
# **Objective:**
# 
# Modify the existing Pydantic models to handle weekly meal planning instead of single meals.
# 
# **Create new Pydantic models that can handle:**
# 
# * Multiple meals across a week.
# * Different meal types (breakfast, lunch, dinner).
# * Weekly budget distribution.
# * Bulk shopping optimization.
# 
# **Requirements:**
# 
# 1. Create a WeeklyMealPlan model
# 2. Create a MealType enum (breakfast, lunch, dinner)
# 3. Modify GroceryShoppingPlan to handle weekly shopping
# 4. Test with a week's worth of meals
# 

# In[ ]:


# from enum import Enum
# from typing import Dict

# # TODO: Create MealType enum
# class MealType(str, Enum):
#     # Add meal types here
#     pass

# # TODO: Create WeeklyMealPlan model
# class WeeklyMealPlan(BaseModel):
#     # Add fields for weekly planning
#     pass

# # TODO: Create WeeklyGroceryPlan model
# class WeeklyGroceryPlan(BaseModel):
#     # Extend GroceryShoppingPlan for weekly use
#     pass

# # TODO: Test your models

from enum import Enum
from typing import Dict

class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch" 
    DINNER = "dinner"
    SNACK = "snack"

class DailyMeals(BaseModel):
    """Meals for one day"""
    date: str = Field(description="Date in YYYY-MM-DD format")
    breakfast: Optional[MealPlan] = Field(default=None, description="Breakfast meal plan")
    lunch: Optional[MealPlan] = Field(default=None, description="Lunch meal plan") 
    dinner: Optional[MealPlan] = Field(default=None, description="Dinner meal plan")
    snacks: Optional[List[MealPlan]] = Field(default=None, description="Snack meal plans")
class WeeklyMealPlan(BaseModel):
    """Complete weekly meal planning"""
    week_start_date: str = Field(description="Start date of the week")
    daily_meals: List[DailyMeals] = Field(description="Meals for each day")
    weekly_themes: List[str] = Field(description="Cooking themes for the week")
    prep_suggestions: List[str] = Field(description="Meal prep recommendations")

class WeeklyGroceryPlan(BaseModel):
    """Weekly grocery shopping strategy"""
    weekly_budget: str = Field(description="Total weekly budget")
    meal_plans: List[DailyMeals] = Field(description="All weekly meals")
    shopping_sections: List[ShoppingCategory] = Field(description="Organized by store sections")
    bulk_items: List[GroceryItem] = Field(description="Items to buy in bulk")
    shopping_tips: List[str] = Field(description="Weekly shopping efficiency tips")
    budget_breakdown: Dict[str, str] = Field(description="Daily budget allocation")

# Test the models
sample_weekly_plan = WeeklyMealPlan(
    week_start_date="2024-01-15",
    daily_meals=[
        DailyMeals(
            date="2024-01-15",
            breakfast=MealPlan(meal_name="Oatmeal", difficulty_level="Easy", servings=2, researched_ingredients=["oats", "milk", "berries"]),
            lunch=MealPlan(meal_name="Salad", difficulty_level="Easy", servings=2, researched_ingredients=["lettuce", "tomatoes", "dressing"]),
            dinner=MealPlan(meal_name="Pasta", difficulty_level="Medium", servings=2, researched_ingredients=["pasta", "sauce", "cheese"])
        )
    ],
    weekly_themes=["Italian Monday", "Taco Tuesday"],
    prep_suggestions=["Wash vegetables on Sunday", "Cook grains in bulk"]
)

display(JSON(sample_weekly_plan.model_dump()))


# <details>
#     <summary>Click here for the solution</summary>
# 
# ```python
# from enum import Enum
# from typing import Dict
# 
# class MealType(str, Enum):
#     BREAKFAST = "breakfast"
#     LUNCH = "lunch" 
#     DINNER = "dinner"
#     SNACK = "snack"
# 
# class DailyMeals(BaseModel):
#     """Meals for one day"""
#     date: str = Field(description="Date in YYYY-MM-DD format")
#     breakfast: Optional[MealPlan] = Field(default=None, description="Breakfast meal plan")
#     lunch: Optional[MealPlan] = Field(default=None, description="Lunch meal plan") 
#     dinner: Optional[MealPlan] = Field(default=None, description="Dinner meal plan")
#     snacks: Optional[List[MealPlan]] = Field(default=None, description="Snack meal plans")
# class WeeklyMealPlan(BaseModel):
#     """Complete weekly meal planning"""
#     week_start_date: str = Field(description="Start date of the week")
#     daily_meals: List[DailyMeals] = Field(description="Meals for each day")
#     weekly_themes: List[str] = Field(description="Cooking themes for the week")
#     prep_suggestions: List[str] = Field(description="Meal prep recommendations")
# 
# class WeeklyGroceryPlan(BaseModel):
#     """Weekly grocery shopping strategy"""
#     weekly_budget: str = Field(description="Total weekly budget")
#     meal_plans: List[DailyMeals] = Field(description="All weekly meals")
#     shopping_sections: List[ShoppingCategory] = Field(description="Organized by store sections")
#     bulk_items: List[GroceryItem] = Field(description="Items to buy in bulk")
#     shopping_tips: List[str] = Field(description="Weekly shopping efficiency tips")
#     budget_breakdown: Dict[str, str] = Field(description="Daily budget allocation")
# 
# # Test the models
# sample_weekly_plan = WeeklyMealPlan(
#     week_start_date="2024-01-15",
#     daily_meals=[
#         DailyMeals(
#             date="2024-01-15",
#             breakfast=MealPlan(meal_name="Oatmeal", difficulty_level="Easy", servings=2, researched_ingredients=["oats", "milk", "berries"]),
#             lunch=MealPlan(meal_name="Salad", difficulty_level="Easy", servings=2, researched_ingredients=["lettuce", "tomatoes", "dressing"]),
#             dinner=MealPlan(meal_name="Pasta", difficulty_level="Medium", servings=2, researched_ingredients=["pasta", "sauce", "cheese"])
#         )
#     ],
#     weekly_themes=["Italian Monday", "Taco Tuesday"],
#     prep_suggestions=["Wash vegetables on Sunday", "Cook grains in bulk"]
# )
# 
# display(JSON(sample_weekly_plan.model_dump()))
# ```
# </details>
# 

# ## Authors
# 

# [Karan Goswami](https://author.skills.network/instructors/karan_goswami)
# 

# ### Other Contributors
# 

# [Tenzin Migmar](https://author.skills.network/instructors/tenzin_migmar)
# 
# [Jigisha Barbhaya](https://author.skills.network/instructors/jigisha_barbhaya)
# 
# [Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)
# 

# ## Change Log
# 
# <details>
#     <summary>Click here for the changelog</summary>
# 
# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2025-07-25|0.1|Karan Goswami|Initial version created|
# |2025-07-26|0.2|Steve Ryan|ID review and format fixes|
# |2025-07-28|0.3|Leah Hanson|QA review and grammar fixes|
# |2025-10-13|0.4|Sathya Priya|Updated the code|
# 
# </details>
# 
# ---
# 
# 
# Copyright © IBM Corporation. All rights reserved.
# 

# In[ ]:




