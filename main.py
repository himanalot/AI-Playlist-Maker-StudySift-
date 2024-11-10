import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests  # For GPT-4o Mini API requests
import json
import random  # For random sampling
#hello
import concurrent.futures  # For ThreadPoolExecutor
import time  # For measuring runtime
import streamlit as st
import threading


# --------------------------- Configuration --------------------------- #

# Spotify API credentials (replace with your actual credentials)

SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# GPT-4o Mini API credentials and endpoint (replace with your actual API key)
#h
GPT4_MINI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
GPT4_MINI_API_KEY = st.secrets["GPT4_MINI_API_KEY"]
client_id = st.secrets["SPOTIPY_CLIENT_ID"]
client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]
# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
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
            st.write("No choices in GPT-4o Mini response.")
            st.write("Response:", result)
            return None
        # Extract the assistant's message content
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.HTTPError as http_err:
        st.write(f"HTTP error occurred: {http_err}")  # HTTP error
        if 'response' in locals():
            st.write(f"Response Text: {response.text}")  # Debugging line
    except json.JSONDecodeError as json_err:
        st.write(f"JSON decode error: {json_err}")  # JSON error
        if 'response' in locals():
            st.write(f"Response Text: {response.text}")  # Print raw response
    except Exception as err:
        st.write(f"Other error occurred: {err}")  # Other errors
        if 'response' in locals():
            st.write(f"Response Text: {response.text}")  # Print raw response
    return None

def get_feature_definitions():
    """
    Returns a string containing the definitions of Spotify audio features.
    """
    feature_definitions = """
- **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
- **danceability**: Describes how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity.
- **energy**: Measure from 0.0 to 1.0 representing intensity and activity.
- **instrumentalness**: Predicts whether a track contains no vocals.
- **liveness**: Detects the presence of an audience in the recording.
- **loudness**: Overall loudness of a track in decibels (dB).
- **speechiness**: Detects the presence of spoken words.
- **valence**: Measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
- **tempo**: The overall estimated tempo of a track in beats per minute (BPM).
"""
    return feature_definitions

def construct_prompt(batch_df, diagnostic, feature_definitions):
    """
    Constructs a prompt for GPT-4o Mini with the song data, the diagnostic, and feature definitions.
    """
    prompt = f"Based on the following user preferences:\n\n'{diagnostic}'\n\n"
    prompt += "Here are the definitions of the audio features:\n\n"
    prompt += feature_definitions + "\n\n"
    prompt += (
        "Please analyze the following songs and decide whether each song should be included in the playlist. "
        "For each song, provide a decision 'Keep' or 'Discard'. Here are the songs:\n\n"
    )

    songs_list = []
    for index, row in batch_df.iterrows():
        song_info = {
            "id": row["id"],
            "name": row["name"],
            "artists": row["artists"],
            "features": {
                "danceability": row["danceability"],
                "energy": row["energy"],
                "valence": row["valence"],
                "tempo": row["tempo"],
                "acousticness": row["acousticness"],
                "instrumentalness": row["instrumentalness"],
                "liveness": row["liveness"],
                "loudness": row["loudness"],
                "speechiness": row["speechiness"],
            },
        }
        songs_list.append(song_info)

    # Convert songs_list to JSON with double quotes
    songs_json = json.dumps(songs_list, indent=2)
    prompt += songs_json + "\n\n"

    prompt += (
        "Respond ONLY with a JSON array of decisions in the following format, "
        "ensuring all property names and string values are enclosed in double quotes, "
        "and without any code fences, explanations, or additional text:\n\n"
        '[\n  {"id": "song_id", "decision": "Keep" or "Discard"},\n  ...\n]\n'
    )

    return prompt

def parse_model_response(response):
    """
    Parses the GPT-4o Mini response to extract song IDs to keep.
    """
    try:
        response = response.strip()
        # Remove code fences if present
        if response.startswith("```") and response.endswith("```"):
            # Remove the starting and ending triple backticks
            response = response[3:-3].strip()
            # Remove optional language identifier after the opening backticks
            if response.startswith('json'):
                response = response[4:].strip()
        decisions = json.loads(response)
        ids_to_keep = [item['id'] for item in decisions if item['decision'].lower() == 'keep']
        return ids_to_keep
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        st.write(f"Failed to parse GPT-4o Mini response: {e}")
        st.write("Raw GPT-4o Mini response:", response)
        return []

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
            st.write(f"Error fetching track info for batch starting at index {start}: {e}")
    
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
            st.write(f"Error fetching audio features for batch starting at index {start}: {e}")
    
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
        st.write(f"Error creating playlist: {e}")
        return None

def filter_songs_with_model(df, diagnostic, feature_definitions):
    """
    Filters songs using GPT-4o Mini based on the song features and the user's diagnostic.
    """
    filtered_ids = []
    
    batch_size = 7  # Adjusted batch size as per your request
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
    
        # Construct prompt
        prompt = construct_prompt(batch_df, diagnostic, feature_definitions)
    
        # Call GPT-4o Mini
        response = call_gpt4_mini(prompt)
    
        if response is None:
            continue  # Skip this batch if there's an error
    
        # Parse response
        ids_to_keep = parse_model_response(response)
        filtered_ids.extend(ids_to_keep)
    
    return filtered_ids

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
                # Remove the starting and ending triple backticks
                response = response[3:-3].strip()
                # Remove optional language identifier after the opening backticks
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
            st.write(f"Error parsing GPT-4o Mini response for search parameters: {e}")
            st.write("Raw GPT-4o Mini response:", response)
            return [genre], []
    else:
        return [genre], []

def filter_playlists_with_model(playlists_info, diagnostic):
    """
    Filters playlists using GPT-4o Mini based on the playlist names and the user's diagnostic.
    Returns a list of playlists to include.
    """
    # Construct the prompt
    playlist_names = [info[1] for info in playlists_info]
    prompt = f"""
Based on the following user preferences:
'{diagnostic}'

Here is a list of playlist names:
{json.dumps(playlist_names, indent=2)}

Please analyze the playlist names and decide whether each playlist is relevant to the user's preferences.
Respond ONLY with a JSON array of decisions in the following format, ensuring all property names and string values are enclosed in double quotes, and without any code fences, explanations, or additional text:

[
  {{"name": "playlist_name", "decision": "Include" or "Exclude"}},
  ...
]
"""
    response = call_gpt4_mini(prompt)
    if response:
        try:
            response = response.strip()
            # Remove code fences if present
            if response.startswith("```") and response.endswith("```"):
                response = response[3:-3].strip()
                if response.startswith('json'):
                    response = response[4:].strip()
            decisions = json.loads(response)
            included_playlists = []
            for decision in decisions:
                name = decision.get('name')
                decision_value = decision.get('decision')
                if decision_value.lower() == 'include':
                    for info in playlists_info:
                        if info[1] == name:
                            included_playlists.append(info)
                            break
            return included_playlists
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            st.write(f"Failed to parse GPT-4o Mini response: {e}")
            st.write("Raw GPT-4o Mini response:", response)
            return playlists_info  # If parsing fails, include all playlists
    else:
        return playlists_info  # If GPT call fails, include all playlists

def search_and_combine_playlists(
    search_genres,
    search_artists,
    existing_playlist_ids,
    limit=200,
    sample_size_per_playlist=10,
    diagnostic=""
):
    """
    Searches for playlists based on genres and artists, combines them with existing playlists.
    Ensures that the same playlist is not processed more than once.
    Returns a list of unique track IDs limited to a specified count.
    """
    track_ids = set()
    processed_playlists = set()  # Set to keep track of processed playlist IDs
    lock = threading.Lock()
    
    def process_playlist(playlist_id, playlist_name, playlist_size):
        # Check playlist size against the specified criteria
        if playlist_size < 100 or playlist_size > 1500:
            # st.write(f"Skipping playlist '{playlist_name}' (has {playlist_size} tracks).")
            return
        # st.write(f"Fetching tracks from playlist: {playlist_name}")
        fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist)
        # st.write(f"Fetched {len(fetched_track_ids)} tracks from playlist: {playlist_name}")
        with lock:
            track_ids.update(fetched_track_ids)
    
    # Function to search playlists and collect playlist info
    def search_playlists(query):
        playlists_info = []
        search_results = sp.search(q=query, type='playlist', limit=10)
        playlist_items = search_results['playlists']['items']
        for playlist in playlist_items:
            playlist_id = playlist['id']
            with lock:
                if playlist_id in processed_playlists:
                    # st.write(f"Skipping already processed playlist '{playlist['name']}' (ID: {playlist_id}).")
                    continue
                processed_playlists.add(playlist_id)
            playlist_name = playlist['name']
            playlist_size = playlist['tracks']['total']
            playlists_info.append((playlist_id, playlist_name, playlist_size))
        return playlists_info
    
    playlists_info = []
    
    # Search for playlists matching the genres
    # st.write("Searching for playlists based on genres...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(search_playlists, f'genre:"{genre}"'): genre for genre in search_genres}
        for future in concurrent.futures.as_completed(futures):
            genre = futures[future]
            try:
                result = future.result()
                playlists_info.extend(result)
            except Exception as e:
                st.write(f"Error searching playlists for genre '{genre}': {e}")
    
    # Search for playlists matching the artists
    # st.write("Searching for playlists based on artists...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(search_playlists, f'artist:"{artist}"'): artist for artist in search_artists}
        for future in concurrent.futures.as_completed(futures):
            artist = futures[future]
            try:
                result = future.result()
                playlists_info.extend(result)
            except Exception as e:
                st.write(f"Error searching playlists for artist '{artist}': {e}")
    
    # Include existing playlists
    if existing_playlist_ids:
        # st.write("Including tracks from existing playlists...")
        for playlist_id in existing_playlist_ids:
            with lock:
                if playlist_id in processed_playlists:
                    # st.write(f"Skipping already processed existing playlist (ID: {playlist_id}).")
                    continue
                processed_playlists.add(playlist_id)
            # Get playlist size and name
            try:
                playlist = sp.playlist(playlist_id, fields='tracks.total,name')
                playlist_size = playlist['tracks']['total']
                playlist_name = playlist['name']
                playlists_info.append((playlist_id, playlist_name, playlist_size))
            except Exception as e:
                st.write(f"Error fetching playlist {playlist_id}: {e}")

    # **Filter playlists using GPT-4o Mini**
    # st.write("\nFiltering playlists based on their names using GPT-4o Mini...")
    playlists_info = filter_playlists_with_model(playlists_info, diagnostic)
    # st.write(f"Playlists after filtering: {len(playlists_info)}")

    # Now process playlists concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_playlist, pid, pname, psize) for pid, pname, psize in playlists_info]
        concurrent.futures.wait(futures)

    # Return only up to the specified limit
    track_ids = list(track_ids)
    if len(track_ids) > limit:
        track_ids = random.sample(track_ids, limit)
    return track_ids

def fetch_playlist_tracks(playlist_id, sample_size=10):
    """
    Efficiently fetches all track IDs from a given playlist and selects a random sample.
    """
    track_ids = []
    try:
        # Get the total number of tracks in the playlist
        total_tracks = sp.playlist_items(playlist_id, fields='total')['total']
        # st.write(f"Total tracks in playlist: {total_tracks}")

        # Fetch all track IDs using pagination
        batch_size = 100  # Maximum allowed by Spotify

        offsets = range(0, total_tracks, batch_size)

        def fetch_batch(offset):
            results = sp.playlist_items(
                playlist_id,
                fields='items.track.id',
                additional_types=['track'],
                limit=batch_size,
                offset=offset
            )
            items = results['items']
            batch_track_ids = []
            for item in items:
                track = item['track']
                if track and track['id']:
                    batch_track_ids.append(track['id'])
            return batch_track_ids

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_offset = {executor.submit(fetch_batch, offset): offset for offset in offsets}
            for future in concurrent.futures.as_completed(future_to_offset):
                try:
                    batch_track_ids = future.result()
                    track_ids.extend(batch_track_ids)
                except Exception as e:
                    offset = future_to_offset[future]
                    st.write(f"Error fetching tracks at offset {offset} from playlist {playlist_id}: {e}")

        # Randomly select sample_size track IDs
        if len(track_ids) >= sample_size:
            track_ids = random.sample(track_ids, sample_size)
        else:
            # st.write(f"Playlist {playlist_id} has fewer tracks than the sample size. Using all available tracks.")
            pass
    except spotipy.exceptions.SpotifyException as e:
        st.write(f"Error fetching playlist {playlist_id}: {e}")
    except Exception as e:
        st.write(f"An unexpected error occurred while fetching playlist {playlist_id}: {e}")
    return track_ids

# --------------------------- Main Functionality --------------------------- #

def main():
    st.title("Automated Playlist Generator")

    # Collect user inputs
    genre = st.text_input("Preferred Genre (e.g., Rock, Pop):")
    mood = st.text_input("Desired Mood (e.g., Calm, Happy, Sad):")
    energy_level = st.selectbox("Energy Level:", ["Low", "Medium", "High"])
    additional_info = st.text_input("Additional Info or Preferences (optional):")
    exclude_genres_input = st.text_input("Genres to Exclude (separated by commas, optional):")
    exclude_artists_input = st.text_input("Artists to Exclude (separated by commas, optional):")
    include_playlists = st.checkbox("Include Existing Playlists for Analysis")

    if st.button("Submit"):
        if not genre or not mood or not energy_level:
            st.error("Please fill in the required fields: Genre, Mood, and Energy Level.")
            return

        exclude_genres = [g.strip() for g in exclude_genres_input.split(',')] if exclude_genres_input else []
        exclude_artists = [a.strip() for a in exclude_artists_input.split(',')] if exclude_artists_input else []

        # Determine search parameters using GPT-4o Mini
        ai_genres, ai_artists = determine_search_parameters(genre, mood, energy_level, additional_info, exclude_genres, exclude_artists)
        if not ai_genres and not ai_artists:
            st.error("No genres or artists suggested by GPT-4o Mini for searching.")
            return

        # Use AI-suggested genres and artists directly
        selected_genres = ai_genres
        selected_artists = ai_artists

        if not selected_genres and not selected_artists:
            st.error("No genres or artists available for searching.")
            return

        # Collect existing playlists if included
        existing_playlists = []
        if include_playlists:
            playlists = sp.current_user_playlists(limit=50)
            playlist_options = {f"{playlist['name']} (ID: {playlist['id']})": playlist['id'] for playlist in playlists['items']}
            selected_playlists = st.multiselect("Select Playlists to Include:", list(playlist_options.keys()))
            existing_playlists = [playlist_options[playlist] for playlist in selected_playlists]

        # Start processing
        process_playlist_generation(
            genre, mood, energy_level, additional_info,
            existing_playlists, exclude_genres, exclude_artists,
            selected_genres, selected_artists
        )

def process_playlist_generation(
    genre,
    mood,
    energy_level,
    additional_info,
    existing_playlists,
    exclude_genres,
    exclude_artists,
    selected_genres,
    selected_artists
):
    """
    Handles the entire process of generating the playlist.
    """
    start_time = time.time()  # Record start time at the very beginning

    status_placeholder = st.empty()  # Placeholder for status messages

    try:
        # Construct diagnostic from user inputs
        diagnostic = f"A playlist with {energy_level.lower()} energy, {mood.lower()} mood, in the {genre} genre."
        if additional_info:
            diagnostic += f" Additional info: {additional_info}"

        # Step 2: Determine Search Parameters using GPT-4o Mini (already done)
        # search_genres and search_artists are already selected by the user

        # Step 3: Search and Combine Playlists with a limit
        with st.spinner('Searching and combining playlists...'):
            
            combined_track_ids = search_and_combine_playlists(
                selected_genres,
                selected_artists,
                existing_playlists,
                limit=200,
                sample_size_per_playlist=10,
                diagnostic=diagnostic  # Pass the diagnostic to filter playlists
            )
        

        if not combined_track_ids:
            st.info("No tracks found based on the specified genres and artists.")
            return

        # Step 4: Retrieve Audio Features
        with st.spinner('Retrieving audio features for collected tracks...'):
            
            audio_features = get_audio_features_concurrent(combined_track_ids)
        status_placeholder.text(f"Retrieved audio features for {len(audio_features)} tracks.")

        # Step 5: Retrieve Track Info
        with st.spinner('Retrieving track information...'):
            
            track_info_list = get_track_info_concurrent(combined_track_ids, exclude_artists)
        status_placeholder.text(f"Retrieved track information for {len(track_info_list)} tracks.")

        # Update combined_track_ids to only include tracks we have info for (excluding tracks by excluded artists)
        combined_track_ids = [track['id'] for track in track_info_list]

        if not combined_track_ids:
            st.info("No tracks available after excluding specified artists.")
            return

        # Step 6: Merge DataFrames
        with st.spinner('Merging track information and audio features...'):
            
            df_features = pd.DataFrame(audio_features)
            df_track_info = pd.DataFrame(track_info_list)
            df = pd.merge(df_track_info, df_features, on='id')
        status_placeholder.text(f"Merged data contains {len(df)} tracks.")

        # Handle missing features by setting to NaN
        features_to_handle = [
            'acousticness',
            'danceability',
            'energy',
            'instrumentalness',
            'liveness',
            'speechiness',
            'valence',
            'tempo',
            'loudness'
        ]
        for feature in features_to_handle:
            if feature in df.columns:
                df[feature] = df[feature].astype(float)  # Ensure correct data type
                df[feature] = df[feature].fillna(float('nan'))

        # Step 7: Apply Filtering with GPT-4o Mini
        with st.spinner('Applying filtering with GPT-4o Mini...'):
            
            feature_definitions = get_feature_definitions()
            filtered_ids = filter_songs_with_model(df, diagnostic, feature_definitions)
        status_placeholder.text(f"Tracks after filtering: {len(filtered_ids)}")

        if not filtered_ids:
            st.info("No tracks match the specified criteria after filtering.")
            return

        # Step 8: Create New Playlist
        with st.spinner('Creating new playlist...'):
            
            playlist_name = f"{genre} - {mood} - {energy_level} Energy"
            new_playlist_url = create_new_playlist(playlist_name, filtered_ids)

        # After creating the new playlist
        if new_playlist_url:
            total_time = time.time() - start_time  # Calculate total time
            status_placeholder.success(f"New Playlist Created: [Open Playlist]({new_playlist_url})\nTotal time taken: {total_time:.2f} seconds")
        else:
            st.error("Failed to create the new playlist.")

    except Exception as e:
        total_time = time.time() - start_time  # Calculate time up to exception
        st.write(f"An error occurred during playlist generation: {e}")
        st.error(f"An error occurred: {e}\nTotal time before error: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()