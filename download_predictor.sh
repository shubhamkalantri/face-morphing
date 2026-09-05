set -euo pipefail

URL="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
DEST="$DIR/shape_predictor_68_face_landmarks.dat"

if [ -f "$DEST" ]; then
    echo "already present: $DEST"
    exit 0
fi

mkdir -p "$DIR"
echo "downloading $URL"
curl -fL "$URL" -o "$DEST.bz2"
bunzip2 "$DEST.bz2"
echo "extracted to $DEST"
