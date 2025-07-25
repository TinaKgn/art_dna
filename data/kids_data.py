
import pandas as pd

kids_data = [
    {
        "genre": "Abstractionism",
        "description": "🎨 Art using shapes and colors instead of real things! Artists play with blocks of color and fun patterns to show feelings and ideas.",
        "time_period": "Early 1900s",
        "key_artists": '["Kandinsky (color wizard)", "Mondrian (loved squares)", "Malevich (shape master)"]',
        "visual_elements": '["Bright blocks of color", "Simple shapes", "Fun patterns", "Lots of imagination"]',
        "philosophy": "Using colors and shapes to share feelings and ideas!"
    },
    {
        "genre": "Art Nouveau",
        "description": "🌸 Fancy art with flowy lines and flowers all around! It’s like nature and art holding hands to make pretty designs.",
        "time_period": "Late 1800s - Early 1900s",
        "key_artists": '["Alphonse Mucha (flower artist)", "Gustav Klimt (golden painter)", "Hector Guimard (cool buildings)"]',
        "visual_elements": '["Flowy lines", "Flowers and leaves", "Swirly shapes", "Pretty decorations"]',
        "philosophy": "Making everyday things look beautiful and special."
    },
    {
        "genre": "Baroque",
        "description": "✨ Super fancy and dramatic art with lots of shiny details and movement, like a big exciting story in pictures.",
        "time_period": "1600s to 1700s",
        "key_artists": '["Caravaggio (drama king)", "Rubens (color master)", "Bernini (sculpture genius)"]',
        "visual_elements": '["Shiny gold", "Strong light and dark", "Moving figures", "Lots of details"]',
        "philosophy": "Making art that feels like a big, exciting adventure!"
    },
    {
        "genre": "Byzantine Art",
        "description": "🕌 Old-timey religious pictures with gold backgrounds and important looking people who don’t move much.",
        "time_period": "300s to 1400s",
        "key_artists": '["Unknown Byzantine artists", "Saint Luke (legend says he painted some)"]',
        "visual_elements": '["Gold backgrounds", "Flat people", "Lots of symbols", "Mosaics"]',
        "philosophy": "Showing stories from religion to inspire people."
    },
    {
        "genre": "Cubism",
        "description": "🧩 Art that breaks things into puzzle pieces and shows them from lots of sides at once! It’s like a fun shape game.",
        "time_period": "Early 1900s",
        "key_artists": '["Picasso (shape breaker)", "Braque (puzzle artist)", "Juan Gris"]',
        "visual_elements": '["Puzzle pieces", "Lots of shapes", "Different viewpoints", "Simple colors"]',
        "philosophy": "Showing things from many sides at the same time!"
    },
    {
        "genre": "Expressionism",
        "description": "🎭 Art that shows BIG feelings! Artists used wild colors and wonky shapes to paint how they felt inside - happy, sad, angry, or scared - instead of making things look real.",
        "time_period": "Early 1900s",
        "key_artists": '["Kandinsky (loved colors!)", "Munch (painted The Scream)", "Kirchner"]',
        "visual_elements": '["Bright, crazy colors", "Wobbly, stretched shapes", "Thick paint blobs", "Zigzag lines"]',
        "philosophy": "Painting feelings, not just what you see! 🌈"
    },
    {
        "genre": "Impressionism",
        "description": "🌞 Paintings that look like a snapshot of a sunny day! Artists use light colors and quick strokes to show how light changes.",
        "time_period": "Late 1800s",
        "key_artists": '["Monet (light master)", "Renoir (people painter)", "Degas (dance artist)"]',
        "visual_elements": '["Light brush strokes", "Soft colors", "Outdoor scenes", "Shiny light spots"]',
        "philosophy": "Capturing moments as they happen!"
    },
    {
        "genre": "Mannerism",
        "description": "🖌️ Art with tall, twisty people doing cool poses! Colors are sometimes wild and the pictures look fancy and a little strange.",
        "time_period": "1500s",
        "key_artists": '["El Greco (stretchy figures)", "Parmigianino (pose master)", "Pontormo"]',
        "visual_elements": '["Tall, twisty people", "Fancy poses", "Bright colors", "Strange perspectives"]',
        "philosophy": "Making art that’s elegant and dramatic!"
    },
    {
        "genre": "Muralism",
        "description": "🎨 Huge wall paintings with bold pictures that tell stories about people and their lives, often about fairness and community.",
        "time_period": "1900s",
        "key_artists": '["Diego Rivera (wall painter)", "José Orozco (storyteller)", "David Siqueiros (bold artist)"]',
        "visual_elements": '["Big pictures", "Strong outlines", "People and stories", "Bright colors"]',
        "philosophy": "Using art to share important messages for everyone."
    },
    {
        "genre": "Neoplasticism",
        "description": "🟦 Art using just squares, rectangles, and basic colors like red, blue, and yellow to make neat, balanced pictures.",
        "time_period": "1910s to 1930s",
        "key_artists": '["Mondrian (square lover)", "Van Doesburg"]',
        "visual_elements": '["Squares and rectangles", "Red, blue, yellow", "Straight lines", "Balanced look"]',
        "philosophy": "Making simple shapes and colors that feel just right."
    },
    {
        "genre": "Pop Art",
        "description": "🎉 Art with bright colors and pictures from comics and ads, like fun cartoons and famous things you see every day.",
        "time_period": "1950s to 1970s",
        "key_artists": '["Andy Warhol (pop king)", "Roy Lichtenstein (comic artist)", "Claes Oldenburg"]',
        "visual_elements": '["Bright colors", "Bold lines", "Repetitions", "Famous images"]',
        "philosophy": "Making everyday things into cool art!"
    },
    {
        "genre": "Primitivism",
        "description": "🌿 Art inspired by old tribal and simple art, using bold colors and shapes to tell stories from long ago.",
        "time_period": "Late 1800s - Early 1900s",
        "key_artists": '["Gauguin (tropical painter)", "Picasso", "Matisse"]',
        "visual_elements": '["Simple shapes", "Bright colors", "Symbols", "Rough textures"]',
        "philosophy": "Going back to simple and powerful art styles."
    },
    {
        "genre": "Realism",
        "description": "🖼️ Art that shows people and places just like they are, everyday stuff without making it look more perfect than real.",
        "time_period": "1800s",
        "key_artists": '["Courbet (real life painter)", "Millet (farm scenes)", "Manet"]',
        "visual_elements": '["Real details", "Natural colors", "Everyday scenes", "True-to-life"]',
        "philosophy": "Showing the world as it really is."
    },
    {
        "genre": "Renaissance",
        "description": "🎨 Art from long ago where artists painted perfect people and scenes using math and perspective, making pictures look real.",
        "time_period": "1400s to 1500s",
        "key_artists": '["Leonardo da Vinci (genius)", "Michelangelo (sculptor)", "Raphael (master painter)"]',
        "visual_elements": '["Perfect shapes", "3D perspective", "Realistic people", "Light and shadows"]',
        "philosophy": "Bringing back old ideas of beauty and balance."
    },
    {
        "genre": "Romanticism",
        "description": "🌄 Paintings full of big feelings and wild nature, like storms and mountains, showing adventure and imagination.",
        "time_period": "1800s",
        "key_artists": '["Caspar David Friedrich (nature lover)", "Delacroix (drama painter)", "Goya"]',
        "visual_elements": '["Dramatic scenes", "Bright colors", "Wild nature", "Strong emotions"]',
        "philosophy": "Showing big feelings and the power of nature."
    },
    {
        "genre": "Suprematism",
        "description": "⬛ Art made with simple shapes like squares and circles in just a few colors to show pure feelings, not things you see.",
        "time_period": "1910s",
        "key_artists": '["Kazimir Malevich (shape artist)"]',
        "visual_elements": '["Simple shapes", "Few colors", "Flat shapes", "Abstract"]',
        "philosophy": "Art that’s about feeling, not pictures."
    },
    {
        "genre": "Surrealism",
        "description": "🌙 Art full of weird dreams and strange things that don’t make sense, like pictures from your imagination or a funny story.",
        "time_period": "1920s to 1950s",
        "key_artists": '["Salvador Dalí (dream painter)", "René Magritte (mysterious scenes)", "Max Ernst"]',
        "visual_elements": '["Dreamlike scenes", "Strange combinations", "Funny or scary images", "Symbols"]',
        "philosophy": "Showing what’s inside your imagination and dreams."
    },
    {
        "genre": "Symbolism",
        "description": "🔮 Art that uses pictures to tell secret stories or feelings about things you can’t see, like magic or mystery.",
        "time_period": "Late 1800s to early 1900s",
        "key_artists": '["Gustave Moreau", "Odilon Redon", "Pierre Puvis de Chavannes"]',
        "visual_elements": '["Magical themes", "Soft colors", "Mysterious pictures", "Secret symbols"]',
        "philosophy": "Using pictures to tell stories about feelings and magic."
    }
]
df = pd.DataFrame(kids_data)
df.to_csv("kids_data_18_genres.csv", index=False)
print("CSV file 'kids_data_18_genres.csv' has been created!")
