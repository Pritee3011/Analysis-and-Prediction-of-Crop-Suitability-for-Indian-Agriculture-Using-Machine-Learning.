# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np

# Import all functions and global variables from our data_handler
try:
    from data_handler import (
    df_clean, CROP_MODEL, AVAILABLE_CROPS, AVAILABLE_STATES, 
    get_crop_recommendations,  
    run_one_way_anova, run_t_test, run_two_way_anova_state
   )
except Exception as e:
    # This ensures the app doesn't crash if the data_handler fails to load (e.g., FileNotFoundError)
    print(f"🚨 WARNING: Could not import data_handler components. Model/Data loading failed: {e}")
    # Define minimal placeholder variables to prevent immediate crash if data_handler failed
    CROP_MODEL = None
    AVAILABLE_CROPS = []
    AVAILABLE_STATES = []


# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_for_flashing_messages' 


# --- CORE NAVIGATION ROUTES ---

@app.route('/')
def home():
    """Renders the homepage (usually index.html)."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Renders the About page."""
    return render_template('about.html') 

@app.route('/crop-info')
def crop_info():
    """Renders the Crop Information page."""
    return render_template('Crop_info.html') 

@app.route('/help')
def help_page():
    """Renders the Help page."""
    return render_template('help.html')


# --- RECOMMENDATION DASHBOARD (GET) ---

@app.route('/recommendation', methods=['GET'])
def recommendation_page():
    """Renders the main recommendation form page."""
    return render_template(
        'recommendation.html',
        available_crops=AVAILABLE_CROPS,
        available_states=AVAILABLE_STATES
    )


# --- ROUTE 1: CROP PREDICTION (Random Forest Model Input) ---
# Find this route in app.py
@app.route('/recommendation/predict', methods=['POST'])
def handle_prediction():
    """Handles the form submission for crop prediction."""
    
    # Capture inputs for "stickiness"
    prediction_inputs = {
        'N_SOIL': request.form.get('N_SOIL'),
        'P_SOIL': request.form.get('P_SOIL'),
        'K_SOIL': request.form.get('K_SOIL'),
        'ph': request.form.get('ph'),
        'TEMPERATURE': request.form.get('TEMPERATURE'),
        'HUMIDITY': request.form.get('HUMIDITY'),
        'RAINFALL': request.form.get('RAINFALL'),
    }

    try:
        # --- NEW FUNCTION CALL ---
        # Call your new function which returns a list
        recommendations, acc = get_crop_recommendations(prediction_inputs, CROP_MODEL)

        return render_template(
            'recommendation.html', 
            # --- NEW VARIABLE ---
            # Pass the whole list to the template
            predicted_crop_list=recommendations, 
            prediction_inputs=prediction_inputs,
            model_accuracy=CROP_MODEL.accuracy,
            available_crops=AVAILABLE_CROPS,
            available_states=AVAILABLE_STATES
        )

    except Exception as e:
        flash(f"❌ Prediction Error: Please check your inputs. {e}", 'danger')
        return render_template(
            'recommendation.html',
            prediction_inputs=prediction_inputs,
            model_accuracy=CROP_MODEL.accuracy,
            available_crops=AVAILABLE_CROPS,
            available_states=AVAILABLE_STATES
        )


# --- ROUTE 2: ONE-WAY ANOVA (Multiple Crop Comparison) ---

@app.route('/recommendation/anova', methods=['POST'])
def handle_anova():
    """Handles the form submission for One-Way ANOVA."""
    
    # Capture inputs immediately for stickiness
    anova_inputs = {
        'anova_crops': request.form.get('anova_crops', '')
    }

    try:
        crop_string = anova_inputs['anova_crops']
        selected_crops = [c.strip() for c in crop_string.split(',')]
        
        results, plot_base64 = run_one_way_anova(df_clean, selected_crops)
        
        if 'error' in results:
            flash(f"❌ ANOVA Error: {results['error']}", 'danger')
            # RENDER TEMPLATE on error to keep inputs
            return render_template(
                'recommendation.html',
                available_crops=AVAILABLE_CROPS,
                available_states=AVAILABLE_STATES,
                anova_inputs=anova_inputs, # Pass inputs back
                # active_section='anova-multiple' # This variable is not used in the HTML
            )
        else:
            # Flash detailed results
            flash(f"📊 ANOVA for {results['crops']} (P-value: {results['p_value']}): **{results['conclusion']}**", 'info')
            
            # RENDER TEMPLATE DIRECTLY (No Redirect) to show results/plot
            return render_template(
                'recommendation.html',
                anova_results=results,
                anova_plot=plot_base64,
                available_crops=AVAILABLE_CROPS,
                available_states=AVAILABLE_STATES,
                anova_inputs=anova_inputs, # Pass inputs back
                # active_section='anova-multiple' # This variable is not used in the HTML
            )
            
    except Exception as e:
        flash(f"❌ Analysis Error: An error occurred during ANOVA calculation. {e}", 'danger')
        # RENDER TEMPLATE on exception to keep inputs
        return render_template(
            'recommendation.html',
            available_crops=AVAILABLE_CROPS,
            available_states=AVAILABLE_STATES,
            anova_inputs=anova_inputs, # Pass inputs back
            # active_section='anova-multiple' # This variable is not used in the HTML
        )


# --- ROUTE 3: T-TEST (Two Crop Comparison) ---

@app.route('/recommendation/ttest', methods=['POST'])
def handle_ttest():
    """Handles the form submission for the T-Test."""
    
    # Capture inputs immediately for stickiness
    ttest_inputs = {
        'crop1': request.form.get('crop1', ''),
        'crop2': request.form.get('crop2', '')
    }
    
    try:
        crop1 = ttest_inputs['crop1']
        crop2 = ttest_inputs['crop2']
        
        results, plot_base64 = run_t_test(df_clean, crop1, crop2)

        if 'error' in results:
            flash(f"❌ T-Test Error: {results['error']}", 'danger')
            # RENDER TEMPLATE on error to keep inputs
            return render_template(
                'recommendation.html',
                available_crops=AVAILABLE_CROPS,
                available_states=AVAILABLE_STATES,
                ttest_inputs=ttest_inputs, # Pass inputs back
                # active_section='t-test-two-crop' # This variable is not used in the HTML
            )
        else:
            flash(f"⚖️ T-Test Result ({crop1} vs {crop2}, P={results['p_value']}): **{results['conclusion']}**", 'info')
            
            # RENDER TEMPLATE DIRECTLY (No Redirect) to show results/plot
            return render_template(
                'recommendation.html',
                ttest_results=results,
                ttest_plot=plot_base64,
                available_crops=AVAILABLE_CROPS,
                available_states=AVAILABLE_STATES,
                ttest_inputs=ttest_inputs, # Pass inputs back
                # active_section='t-test-two-crop' # This variable is not used in the HTML
            )

    except Exception as e:
        flash(f"❌ Analysis Error: An error occurred during T-Test calculation. {e}", 'danger')
        # RENDER TEMPLATE on exception to keep inputs
        return render_template(
            'recommendation.html',
            available_crops=AVAILABLE_CROPS,
            available_states=AVAILABLE_STATES,
            ttest_inputs=ttest_inputs, # Pass inputs back
            # active_section='t-test-two-crop' # This variable is not used in the HTML
        )


# --- ROUTE 4: TWO-WAY ANOVA (Crop x State Comparison) ---

@app.route('/recommendation/anova_state', methods=['POST'])
def handle_two_way_anova():
    """Handles the form submission for Two-Way ANOVA."""
    
    # Capture inputs immediately for stickiness
    anova_inputs = {
        'anova_state_crops': request.form.get('anova_state_crops', ''),
        'anova_states': request.form.get('anova_states', '')
    }
    
    try:
        crop_string = anova_inputs['anova_state_crops']
        state_string = anova_inputs['anova_states']
        
        selected_crops = [c.strip() for c in crop_string.split(',')]
        selected_states = [s.strip() for s in state_string.split(',')]
        
        results, plot_base64 = run_two_way_anova_state(df_clean, selected_crops, selected_states)

        if 'error' in results:
            flash(f"❌ Two-Way ANOVA Error: {results['error']}", 'danger')
            
            # CRITICAL FIX: Render the template directly on soft error (like 'No data found')
            # to display the flash message AND keep the form filled.
            return render_template(
                'recommendation.html',
                available_crops=AVAILABLE_CROPS,
                available_states=AVAILABLE_STATES,
                anova_inputs=anova_inputs,  # Pass inputs back on error
                # active_section='anova-crop-state' # This variable is not used in the HTML
            )
        else:
            # FIX: Corrected flash message to use the right keys (crop_conclusion, interaction_conclusion)
            flash(f"🌍 Two-Way ANOVA Complete! Crop effect: **{results['crop_conclusion']}** | State effect: **{results['state_conclusion']}** | Interaction: **{results['interaction_conclusion']}**", 'info')
            
            # RENDER TEMPLATE DIRECTLY (No Redirect) to show results/plot
            return render_template(
                'recommendation.html',
                twowayanova_results=results,
                twowayanova_plot=plot_base64,
                available_crops=AVAILABLE_CROPS,
                available_states=AVAILABLE_STATES,
                anova_inputs=anova_inputs,  # Pass inputs back on success
                # active_section='anova-crop-state' # This variable is not used in the HTML
            )

    except Exception as e:
        flash(f"❌ Analysis Error: An error occurred during Two-Way ANOVA calculation. {e}", 'danger')
        
        # On hard exception, render template to keep inputs
        return render_template(
            'recommendation.html',
            available_crops=AVAILABLE_CROPS,
            available_states=AVAILABLE_STATES,
            anova_inputs=anova_inputs, # Pass inputs back on hard error
            # active_section='anova-crop-state' # This variable is not used in the HTML
        )

# --- RUN APP ---
if __name__ == '__main__':
    print("Starting Farm-Forward Flask Server...")
    app.run(debug=True)