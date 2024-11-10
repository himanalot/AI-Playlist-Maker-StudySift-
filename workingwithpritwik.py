import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests  # For GPT-4o Mini API requests
import json
import random  # For random sampling
import threading  # For concurrent API requests
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

# --------------------------- Configuration --------------------------- #

# Spotify API credentials (replace with your actual credentials)
SPOTIPY_CLIENT_ID = 'fe6a043531214ac1abf575878699f986'
SPOTIPY_CLIENT_SECRET = '3ca2ef78541a47109b884f42a90881db'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# GPT-4o Mini API credentials and endpoint (replace with your actual API key)
GPT4_MINI_API_KEY = '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'
GPT4_MINI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'

# --------------------------- Initialization --------------------------- #

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope='playlist-read-private playlist-read-collaborative playlist-modify-public playlist-modify-private'
), requests_timeout=10)  # Set a timeout for API requests

# --------------------------- Helper Functions --------------------------- #

def call_gpt4_mini(prompt):
    """
    Calls the GPT-4o Mini API with the provided prompt and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {GPT4_MINI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GPT4_MINI_API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for bad responses
        result = response.json()
        if 'choices' not in result or not result['choices']:
            print("No choices in GPT-4o Mini response.")
            print("Response:", result)
            return None
        # Extract the assistant's message content
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP error
        if 'response' in locals():
            print(f"Response Text: {response.text}")  # Debugging line
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")  # JSON error
        if 'response' in locals():
            print(f"Response Text: {response.text}")  # Print raw response
    except Exception as err:
        print(f"Other error occurred: {err}")  # Other errors
        if 'response' in locals():
            print(f"Response Text: {response.text}")  # Print raw response
    return None

def get_feature_definitions():
    """
    Returns a string containing the definitions of Spotify audio features.
    """
    feature_definitions = """
- **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- **danceability**: Describes how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity. 0.0 is least danceable, 1.0 is most danceable.
- **energy**: Measure from 0.0 to 1.0 representing intensity and activity. High energy tracks feel fast and loud.
- **instrumentalness**: Predicts whether a track contains no vocals. Values closer to 1.0 indicate a higher likelihood of no vocal content.
- **liveness**: Detects the presence of an audience in the recording. Values above 0.8 indicate a high probability of live performance. Range is 0.0-1.0.
- **loudness**: Overall loudness of a track in decibels (dB). Values typically range between -60 and 0 dB.
- **speechiness**: Detects the presence of spoken words. Values above 0.66 indicate tracks made entirely of spoken words. Range is 0.0-1.0.
- **valence**: Measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. High valence sounds more positive.
- **tempo**: The overall estimated tempo of a track in beats per minute (BPM).
"""
    return feature_definitions

def get_track_info_concurrent(track_ids, exclude_artists):
    """
    Retrieves track information (name, artists) for a list of track IDs using threading for concurrency.
    Excludes tracks by artists specified in exclude_artists.
    """
    track_info_list = []
    lock = threading.Lock()
    exclude_artists_lower = [artist.lower() for artist in exclude_artists]
    
    def fetch_batch(start):
        batch = track_ids[start:start+50]
        try:
            tracks = sp.tracks(batch)['tracks']
            with lock:
                for track in tracks:
                    if track and track['id']:
                        track_artists = [artist['name'] for artist in track['artists']]
                        # Exclude tracks by excluded artists
                        if any(artist.lower() in exclude_artists_lower for artist in track_artists):
                            continue  # Skip this track
                        track_info = {
                            'id': track['id'],
                            'name': track['name'],
                            'artists': ', '.join(track_artists)
                        }
                        track_info_list.append(track_info)
        except Exception as e:
            print(f"Error fetching track info for batch starting at index {start}: {e}")
    
    threads = []
    for i in range(0, len(track_ids), 50):
        thread = threading.Thread(target=fetch_batch, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return track_info_list

def get_audio_features_concurrent(track_ids):
    """
    Retrieves audio features for a list of track IDs using threading for concurrency.
    """
    audio_features = []
    lock = threading.Lock()
    
    def fetch_batch(start):
        batch = track_ids[start:start + 100]
        try:
            features = sp.audio_features(batch)
            with lock:
                audio_features.extend([f for f in features if f is not None])
        except Exception as e:
            print(f"Error fetching audio features for batch starting at index {start}: {e}")
    
    threads = []
    for i in range(0, len(track_ids), 100):
        thread = threading.Thread(target=fetch_batch, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return audio_features

def create_new_playlist(name, track_ids):
    """
    Creates a new Spotify playlist with the specified track IDs.
    """
    try:
        user_id = sp.current_user()['id']
        playlist = sp.user_playlist_create(user=user_id, name=name, public=True, description='Generated by Automated Playlist Generator')
        # Spotify API allows adding up to 100 tracks per request
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            sp.playlist_add_items(playlist_id=playlist['id'], items=batch)
        return playlist['external_urls']['spotify']
    except Exception as e:
        print(f"Error creating playlist: {e}")
        return None

def fetch_playlist_tracks(playlist_id, sample_size=10):
    """
    Efficiently fetches all track IDs from a given playlist and selects a random sample.
    """
    track_ids = []
    try:
        # Get the total number of tracks in the playlist
        total_tracks = sp.playlist_items(playlist_id, fields='total')['total']
        print(f"Total tracks in playlist: {total_tracks}")

        # Fetch all track IDs using pagination
        batch_size = 100  # Maximum allowed by Spotify
        for offset in range(0, total_tracks, batch_size):
            results = sp.playlist_items(
                playlist_id,
                fields='items.track.id',
                additional_types=['track'],
                limit=batch_size,
                offset=offset
            )
            items = results['items']
            for item in items:
                track = item['track']
                if track and track['id']:
                    track_ids.append(track['id'])
        
        # Randomly select sample_size track IDs
        if len(track_ids) >= sample_size:
            track_ids = random.sample(track_ids, sample_size)
        else:
            print(f"Playlist {playlist_id} has fewer tracks than the sample size. Using all available tracks.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error fetching playlist {playlist_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching playlist {playlist_id}: {e}")
    return track_ids

def compute_feature_statistics(df):
    """
    Computes statistical summaries for each audio feature.
    """
    feature_stats = {}
    features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness',
                'instrumentalness', 'liveness', 'loudness', 'speechiness']
    
    for feature in features:
        if feature in df.columns:
            stats = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max()
            }
            feature_stats[feature] = stats
    return feature_stats

def generate_filtering_criteria(diagnostic, feature_definitions, feature_stats):
    """
    Uses GPT-4o Mini to generate filtering criteria based on the diagnostic, feature definitions, and statistical summaries.
    """
    stats_text = "Here are the statistical summaries of the audio features based on the current dataset:\n\n"
    for feature, stats in feature_stats.items():
        stats_text += f"- **{feature}**: mean={stats['mean']:.3f}, std={stats['std']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}\n"
    
    prompt = f"""
Based on the following user preferences:

'{diagnostic}'

Here are the definitions of the audio features:

{feature_definitions}

{stats_text}

Please consider the interdependence of features when generating filtering criteria. Provide appropriate criteria that take into account how features interact with each other to match the user preferences.

Respond ONLY with a valid JSON object in the following format:

{{
  "criteria": [
    {{"condition": "your condition here using features and logical operators"}},
    ...
  ]
}}
"""
    response = call_gpt4_mini(prompt)
    if response:
        try:
            # Remove code fences if present
            response = response.strip()
            if response.startswith("```") and response.endswith("```"):
                response = response[3:-3].strip()
                if response.startswith('json'):
                    response = response[4:].strip()
            # Parse the JSON response
            criteria = json.loads(response)
            return criteria
        except json.JSONDecodeError as e:
            print(f"Error parsing GPT-4o Mini response for filtering criteria: {e}")
            print("Raw GPT-4o Mini response:", response)
            return None
    else:
        return None

def apply_complex_filtering_criteria(df, criteria):
    """
    Applies complex filtering criteria to the DataFrame of songs.
    """
    if 'criteria' not in criteria:
        print("No valid criteria found.")
        return df
    
    for rule in criteria['criteria']:
        condition = rule.get('condition', '')
        
        if not condition:
            continue  # Skip invalid rules
        
        try:
            # Replace any logical operators to match pandas query syntax
            condition = condition.replace('AND', 'and').replace('OR', 'or')
            # Ensure the condition is safe to evaluate
            allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_><=.&|() "
            if not all(c in allowed_chars for c in condition):
                print(f"Unsafe condition detected and skipped: {condition}")
                continue
            df = df.query(condition)
        except Exception as e:
            print(f"Error applying condition '{condition}': {e}")
            continue
    
    return df

def determine_search_parameters(genre, mood, energy_level, additional_info, exclude_genres, exclude_artists):
    """
    Uses GPT-4o Mini to determine which genres and artists to search for based on user inputs.
    Accounts for genres and artists to exclude.
    """
    prompt = f"""
Based on the following user preferences:
- Genre: {genre}
- Mood: {mood}
- Energy Level: {energy_level}
- Additional Info: {additional_info if additional_info else 'None'}
- Excluded Genres: {', '.join(exclude_genres) if exclude_genres else 'None'}
- Excluded Artists: {', '.join(exclude_artists) if exclude_artists else 'None'}

Please provide a list of genres and artists that match these preferences, excluding any genres or artists specified.

**Respond ONLY with valid JSON in the following format, without any code fences or additional text:**

{{
    "genres": ["genre1", "genre2", ...],
    "artists": ["artist1", "artist2", ...]
}}
"""
    response = call_gpt4_mini(prompt)
    if response:
        try:
            # Strip leading and trailing whitespace
            response = response.strip()
            # Remove code fences if present
            if response.startswith("```") and response.endswith("```"):
                response = response[3:-3].strip()
                if response.startswith('json'):
                    response = response[4:].strip()
            search_params = json.loads(response)
            genres = search_params.get('genres', [])
            artists = search_params.get('artists', [])
            # Remove excluded genres and artists (case-insensitive)
            genres = [g for g in genres if g.lower() not in [eg.lower() for eg in exclude_genres]]
            artists = [a for a in artists if a.lower() not in [ea.lower() for ea in exclude_artists]]
            return genres, artists
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing GPT-4o Mini response for search parameters: {e}")
            print("Raw GPT-4o Mini response:", response)
            return [genre], []
    else:
        return [genre], []

def get_user_input_gui(root):
    """
    Collects user inputs for genre, mood, energy level, and additional info via GUI.
    """
    # Create a new window for user inputs
    input_window = tk.Toplevel(root)
    input_window.title("Playlist Preferences")
    
    # Genre
    tk.Label(input_window, text="Preferred Genre (e.g., Rock, Pop):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
    genre_entry = tk.Entry(input_window, width=50)
    genre_entry.grid(row=0, column=1, padx=5, pady=5)
    
    # Mood
    tk.Label(input_window, text="Desired Mood (e.g., Calm, Happy, Sad):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
    mood_entry = tk.Entry(input_window, width=50)
    mood_entry.grid(row=1, column=1, padx=5, pady=5)
    
    # Energy Level
    tk.Label(input_window, text="Energy Level:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
    energy_level_var = tk.StringVar(value="Medium")
    energy_levels = ["Low", "Medium", "High"]
    energy_level_menu = ttk.Combobox(input_window, textvariable=energy_level_var, values=energy_levels, state='readonly')
    energy_level_menu.grid(row=2, column=1, padx=5, pady=5)
    
    # Additional Info
    tk.Label(input_window, text="Additional Info or Preferences (optional):").grid(row=3, column=0, sticky='w', padx=5, pady=5)
    additional_info_entry = tk.Entry(input_window, width=50)
    additional_info_entry.grid(row=3, column=1, padx=5, pady=5)
    
    # Excluded Genres
    tk.Label(input_window, text="Genres to Exclude (separated by commas, optional):").grid(row=4, column=0, sticky='w', padx=5, pady=5)
    exclude_genres_entry = tk.Entry(input_window, width=50)
    exclude_genres_entry.grid(row=4, column=1, padx=5, pady=5)
    
    # Excluded Artists
    tk.Label(input_window, text="Artists to Exclude (separated by commas, optional):").grid(row=5, column=0, sticky='w', padx=5, pady=5)
    exclude_artists_entry = tk.Entry(input_window, width=50)
    exclude_artists_entry.grid(row=5, column=1, padx=5, pady=5)
    
    # Include Existing Playlists
    include_playlists_var = tk.BooleanVar()
    include_playlists_check = tk.Checkbutton(input_window, text="Include Existing Playlists for Analysis", variable=include_playlists_var)
    include_playlists_check.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
    
    # Submit Button
    def on_submit():
        genre = genre_entry.get().strip()
        mood = mood_entry.get().strip()
        energy_level = energy_level_var.get().strip()
        additional_info = additional_info_entry.get().strip()
        exclude_genres_input = exclude_genres_entry.get().strip()
        exclude_artists_input = exclude_artists_entry.get().strip()
        
        exclude_genres = [g.strip() for g in exclude_genres_input.split(',')] if exclude_genres_input else []
        exclude_artists = [a.strip() for a in exclude_artists_input.split(',')] if exclude_artists_input else []
        
        include_playlists = include_playlists_var.get()
        
        if not genre or not mood or not energy_level:
            messagebox.showerror("Error", "Please fill in the required fields: Genre, Mood, and Energy Level.")
            return
        
        input_window.destroy()
        
        # Proceed to deselection window if AI suggestions are available
        ai_genres, ai_artists = determine_search_parameters(genre, mood, energy_level, additional_info, exclude_genres, exclude_artists)
        if not ai_genres and not ai_artists:
            messagebox.showerror("Error", "No genres or artists suggested by GPT-4o Mini for searching.")
            return
        
        selected_genres, selected_artists = get_user_selections_gui(root, ai_genres, ai_artists)
        
        # Collect existing playlists if included
        existing_playlists = []
        if include_playlists:
            existing_playlists = select_existing_playlists_gui(root)
        
        # Start processing in a separate thread to keep the GUI responsive
        processing_thread = threading.Thread(target=process_playlist_generation, args=(
            genre, mood, energy_level, additional_info, existing_playlists, exclude_genres, exclude_artists, selected_genres, selected_artists
        ))
        processing_thread.start()
    
    submit_button = tk.Button(input_window, text="Submit", command=on_submit)
    submit_button.grid(row=7, column=0, columnspan=2, pady=10)
    
    root.wait_window(input_window)
    
    def select_existing_playlists_gui(root):
        """
        Allows the user to select existing playlists for analysis.
        """
        selection_window = tk.Toplevel(root)
        selection_window.title("Select Existing Playlists")
        
        tk.Label(selection_window, text="Select Playlists to Include:").pack(padx=5, pady=5)
        
        playlists = sp.current_user_playlists(limit=50)
        playlist_vars = []
        
        for playlist in playlists['items']:
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(selection_window, text=f"{playlist['name']} (ID: {playlist['id']})", variable=var)
            chk.pack(anchor='w', padx=5)
            playlist_vars.append((playlist['id'], var))
        
        selected_playlists = []
        
        def on_confirm():
            for pid, var in playlist_vars:
                if var.get():
                    selected_playlists.append(pid)
            selection_window.destroy()
        
        confirm_button = tk.Button(selection_window, text="Confirm", command=on_confirm)
        confirm_button.pack(pady=10)
        
        root.wait_window(selection_window)
        
        return selected_playlists

def get_user_selections_gui(root, genres, artists):
    """
    Presents the AI-suggested genres and artists to the user for deselection via GUI.
    Returns the selected genres and artists.
    """
    selection_window = tk.Toplevel(root)
    selection_window.title("Select Genres and Artists to Include")
    
    # Genres Selection
    tk.Label(selection_window, text="Select Genres to Include:").pack(padx=5, pady=5)
    genre_vars = []
    for genre in genres:
        var = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(selection_window, text=genre, variable=var)
        chk.pack(anchor='w', padx=20)
        genre_vars.append((genre, var))
    
    # Artists Selection
    tk.Label(selection_window, text="Select Artists to Include:").pack(padx=5, pady=5)
    artist_vars = []
    for artist in artists:
        var = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(selection_window, text=artist, variable=var)
        chk.pack(anchor='w', padx=20)
        artist_vars.append((artist, var))
    
    selected_genres = []
    selected_artists = []
    
    def on_submit():
        nonlocal selected_genres, selected_artists
        selected_genres = [genre for genre, var in genre_vars if var.get()]
        selected_artists = [artist for artist, var in artist_vars if var.get()]
        if not selected_genres and not selected_artists:
            messagebox.showerror("Error", "You must select at least one genre or artist.")
            return
        selection_window.destroy()
    
    submit_button = tk.Button(selection_window, text="Submit Selections", command=on_submit)
    submit_button.pack(pady=10)
    
    root.wait_window(selection_window)
    
    return selected_genres, selected_artists

def search_and_combine_playlists(search_genres, search_artists, existing_playlist_ids, limit=200, sample_size_per_playlist=10):
    """
    Searches for playlists based on genres and artists, combines them with existing playlists.
    Ensures that the same playlist is not processed more than once.
    Returns a list of unique track IDs limited to a specified count.
    """
    track_ids = set()
    processed_playlists = set()  # Set to keep track of processed playlist IDs
    
    print("Searching for playlists based on genres...")
    # Search for playlists matching the genres
    for genre in search_genres:
        print(f"Searching for playlists with genre: {genre}")
        search_results = sp.search(q=f'genre:"{genre}"', type='playlist', limit=10)
        playlist_items = search_results['playlists']['items']
        if not playlist_items:
            print(f"No playlists found for genre '{genre}'.")
            continue
        selected_playlists = playlist_items  # Use all found playlists

        for playlist in selected_playlists:
            playlist_id = playlist['id']
            # Check if playlist has already been processed
            if playlist_id in processed_playlists:
                print(f"Skipping already processed playlist '{playlist['name']}' (ID: {playlist_id}).")
                continue
            # Get the total number of tracks in the playlist
            playlist_size = playlist['tracks']['total']
            # Check playlist size against the specified criteria
            if playlist_size < 100 or playlist_size > 1500:
                print(f"Skipping playlist '{playlist['name']}' (has {playlist_size} tracks).")
                continue
            print(f"Fetching tracks from playlist: {playlist['name']}")
            fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist)
            print(f"Fetched {len(fetched_track_ids)} tracks from playlist: {playlist['name']}")
            track_ids.update(fetched_track_ids)
            processed_playlists.add(playlist_id)  # Mark this playlist as processed

    print("Searching for playlists based on artists...")
    # Search for playlists matching the artists
    for artist in search_artists:
        print(f"Searching for playlists with artist: {artist}")
        search_results = sp.search(q=f'artist:"{artist}"', type='playlist', limit=10)
        playlist_items = search_results['playlists']['items']
        if not playlist_items:
            print(f"No playlists found for artist '{artist}'.")
            continue
        selected_playlists = playlist_items  # Use all found playlists

        for playlist in selected_playlists:
            playlist_id = playlist['id']
            # Check if playlist has already been processed
            if playlist_id in processed_playlists:
                print(f"Skipping already processed playlist '{playlist['name']}' (ID: {playlist_id}).")
                continue
            # Get the total number of tracks in the playlist
            playlist_size = playlist['tracks']['total']
            # Check playlist size against the specified criteria
            if playlist_size < 100 or playlist_size > 1500:
                print(f"Skipping playlist '{playlist['name']}' (has {playlist_size} tracks).")
                continue
            print(f"Fetching tracks from playlist: {playlist['name']}")
            fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist)
            print(f"Fetched {len(fetched_track_ids)} tracks from playlist: {playlist['name']}")
            track_ids.update(fetched_track_ids)
            processed_playlists.add(playlist_id)  # Mark this playlist as processed

    # Include tracks from existing playlists
    if existing_playlist_ids:
        print("Including tracks from existing playlists...")
        for playlist_id in existing_playlist_ids:
            # Check if playlist has already been processed
            if playlist_id in processed_playlists:
                print(f"Skipping already processed existing playlist (ID: {playlist_id}).")
                continue
            # Get playlist size and name
            playlist = sp.playlist(playlist_id, fields='tracks.total,name')
            playlist_size = playlist['tracks']['total']
            playlist_name = playlist['name']
            if playlist_size < 100 or playlist_size > 1500:
                print(f"Skipping existing playlist '{playlist_name}' (has {playlist_size} tracks).")
                continue
            print(f"Fetching tracks from existing playlist: {playlist_name}")
            fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist)
            print(f"Fetched {len(fetched_track_ids)} tracks from existing playlist: {playlist_name}")
            track_ids.update(fetched_track_ids)
            processed_playlists.add(playlist_id)  # Mark this playlist as processed

    # Return only up to the specified limit
    track_ids = list(track_ids)
    if len(track_ids) > limit:
        track_ids = random.sample(track_ids, limit)
    return track_ids

def process_playlist_generation(genre, mood, energy_level, additional_info, existing_playlists, exclude_genres, exclude_artists, selected_genres, selected_artists):
    """
    Handles the entire process of generating the playlist.
    Runs in a separate thread to keep the GUI responsive.
    """
    try:
        # Construct diagnostic from user inputs
        diagnostic = f"A playlist with {energy_level.lower()} energy, {mood.lower()} mood, in the {genre} genre."
        if additional_info:
            diagnostic += f" Additional info: {additional_info}"
    
        print(f"\nGenres to search: {selected_genres}")
        print(f"Artists to search: {selected_artists}")
    
        # Step 3: Search and Combine Playlists with a limit
        print("\nStarting search and combination of playlists...")
        combined_track_ids = search_and_combine_playlists(selected_genres, selected_artists, existing_playlists, limit=200, sample_size_per_playlist=10)
        print(f"\nTotal unique tracks collected (limited to 200): {len(combined_track_ids)}")
    
        if not combined_track_ids:
            messagebox.showinfo("Info", "No tracks found based on the specified genres and artists.")
            return
    
        # Step 4: Retrieve Audio Features
        print("\nRetrieving audio features for collected tracks...")
        audio_features = get_audio_features_concurrent(combined_track_ids)
        print(f"Retrieved audio features for {len(audio_features)} tracks.")
    
        # Step 5: Retrieve Track Info
        print("\nRetrieving track information...")
        track_info_list = get_track_info_concurrent(combined_track_ids, exclude_artists)
        print(f"Retrieved track information for {len(track_info_list)} tracks.")
    
        # Update combined_track_ids to only include tracks we have info for (excluding tracks by excluded artists)
        combined_track_ids = [track['id'] for track in track_info_list]
    
        if not combined_track_ids:
            messagebox.showinfo("Info", "No tracks available after excluding specified artists.")
            return
    
        # Step 6: Merge DataFrames
        print("\nMerging track information and audio features...")
        df_features = pd.DataFrame(audio_features)
        df_track_info = pd.DataFrame(track_info_list)
        df = pd.merge(df_track_info, df_features, on='id')
        print(f"Merged data contains {len(df)} tracks.")
    
        # Handle missing features by setting to NaN
        features_to_handle = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                              'speechiness', 'valence', 'tempo', 'loudness']
        for feature in features_to_handle:
            if feature in df.columns:
                df[feature] = df[feature].astype(float)  # Ensure correct data type
                df[feature] = df[feature].fillna(float('nan'))
    
        # Step 7: Compute Feature Statistics
        print("\nComputing feature statistics...")
        feature_stats = compute_feature_statistics(df)
        print(f"Feature statistics: {feature_stats}")
    
        # Step 8: Generate Filtering Criteria with GPT-4o Mini
        print("\nGenerating filtering criteria with GPT-4o Mini...")
        feature_definitions = get_feature_definitions()
        criteria = generate_filtering_criteria(diagnostic, feature_definitions, feature_stats)
        if criteria is None:
            messagebox.showerror("Error", "Failed to generate filtering criteria.")
            return
        print(f"Generated filtering criteria: {criteria}")
        
        # Step 9: Apply Complex Filtering Criteria to Songs
        print("\nApplying filtering criteria to songs...")
        df_filtered = apply_complex_filtering_criteria(df, criteria)
        print(f"\nTracks after filtering: {len(df_filtered)}")
        
        if df_filtered.empty:
            messagebox.showinfo("Info", "No tracks match the specified criteria after filtering.")
            return
        
        filtered_ids = df_filtered['id'].tolist()
    
        # Step 10: Create New Playlist
        print("\nCreating new playlist...")
        playlist_name = f"{genre} - {mood} - {energy_level} Energy"
        new_playlist_url = create_new_playlist(playlist_name, filtered_ids)
        if new_playlist_url:
            messagebox.showinfo("Success", f"New Playlist Created: {new_playlist_url}")
        else:
            messagebox.showerror("Error", "Failed to create the new playlist.")
    
    except Exception as e:
        print(f"An error occurred during playlist generation: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

# --------------------------- Main Functionality --------------------------- #

def main():
    # Initialize Tkinter root window
    root = tk.Tk()
    root.title("Automated Playlist Generator")
    root.geometry("600x400")
    
    # Hide the root window as we'll use dialog windows
    root.withdraw()
    
    # Start the process
    get_user_input_gui(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()