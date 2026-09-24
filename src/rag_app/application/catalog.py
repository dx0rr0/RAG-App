from datetime import datetime, timezone

from rag_app.domain.validation import validate_youtube_channel_url


class CatalogService:
    def __init__(self, catalog, media_catalog):
        self.catalog = catalog
        self.media_catalog = media_catalog

    def add_channel(self, url):
        normalized = validate_youtube_channel_url(url)
        existing = self.catalog.get_channel_by_url(normalized)
        channel, videos = self.media_catalog.discover(normalized)
        if existing:
            channel["id"] = existing["id"]
        saved = self.catalog.save_channel(channel)
        videos = [dict(video, channel_id=saved["id"]) for video in videos]
        self.catalog.upsert_videos(videos)
        self.catalog.mark_channel_scanned(channel["id"])
        return self.catalog.get_channel(saved["id"]), self.catalog.list_videos(saved["id"])

    def refresh_channel(self, channel_id):
        channel = self.catalog.get_channel(channel_id)
        if channel is None:
            raise KeyError(channel_id)
        discovered, videos = self.media_catalog.discover(channel["url"])
        videos = [dict(video, channel_id=channel_id) for video in videos]
        self.catalog.upsert_videos(videos)
        self.catalog.mark_channel_scanned(channel_id)
        return self.catalog.get_channel(channel_id), self.catalog.list_videos(channel_id)
