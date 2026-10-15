require "active_support/all"
require "fileutils"
require "json"
require "nokogiri"
require "open-uri"
require "time"

module Helpers
  extend ActiveSupport::NumberHelper
end

module Jekyll
  class GoogleScholarCitationsStore
    CACHE_PATH = ".jekyll-cache/google-scholar-citations.json"
    PUBLISHED_PATH = "assets/data/google-scholar-citations.json"

    def self.for_site(site)
      site.instance_variable_get(:@google_scholar_citations) ||
        site.instance_variable_set(:@google_scholar_citations, new(site))
    end

    def initialize(site)
      @site = site
      @records = {}
      @attempted = {}
      merge_records(site.data["google_scholar_citations"] || {})
      path = File.join(site.source, CACHE_PATH)
      merge_records(JSON.parse(File.read(path))) if File.file?(path)
    rescue JSON::ParserError, SystemCallError => e
      Jekyll.logger.warn "Google Scholar:", "Cannot read saved citations (#{e.message}); using verified seed data."
    end

    def lookup(scholar_id, article_id)
      key = "#{scholar_id}:#{article_id}"
      return @records[key] if @attempted[key]

      @attempted[key] = true
      url = "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=#{scholar_id}&citation_for_view=#{key}"
      begin
        sleep(rand(1.5..3.5))
        doc = URI.open(url, "User-Agent" => "Ruby/#{RUBY_VERSION}", open_timeout: 10, read_timeout: 15) do |response|
          Nokogiri::HTML(response)
        end
        # A missing count or a block page is unknown, never a verified zero.
        raise "Scholar article title missing" if doc.at_css("#gsc_oci_title")&.text.to_s.strip.empty?

        match = doc.css('meta[name="description"], meta[property="og:description"]').filter_map do |meta|
          meta["content"].to_s.match(/Cited by (\d[\d,]*)\b/)
        end.first
        raise "Citation count missing from Scholar page" unless match

        @records[key] = { "count" => match[1].delete(",").to_i, "fetched_at" => Time.now.utc.iso8601 }
      rescue StandardError => e
        fallback = @records[key] ? "keeping count from #{@records[key]['fetched_at']}" : "showing Scholar link only"
        Jekyll.logger.warn "Google Scholar:", "#{key}: #{e.class} - #{e.message}; #{fallback}."
      end
      @records[key]
    end

    def write
      # The published JSON travels with gh-pages; the workflow restores it next time.
      json = JSON.pretty_generate(@records.sort.to_h) + "\n"
      [File.join(@site.source, CACHE_PATH), File.join(@site.dest, PUBLISHED_PATH)].each do |path|
        FileUtils.mkdir_p(File.dirname(path))
        File.write(path, json)
      end
    end

    private

    def merge_records(records)
      return unless records.is_a?(Hash)

      records.each do |key, record|
        next unless record.is_a?(Hash) && record["count"].is_a?(Integer) && record["count"] >= 0

        begin
          fetched_at = Time.iso8601(record.fetch("fetched_at"))
        rescue ArgumentError, TypeError, KeyError
          next
        end
        next if @records[key] && Time.iso8601(@records[key]["fetched_at"]) >= fetched_at

        @records[key] = { "count" => record["count"], "fetched_at" => fetched_at.utc.iso8601 }
      end
    end
  end

  class GoogleScholarCitationsTag < Liquid::Tag
    def initialize(tag_name, params, tokens)
      super
      @scholar_id, @article_id, @field = params.split
    end

    def render(context)
      scholar_id = context[@scholar_id]
      article_id = context[@article_id]
      return "" if scholar_id.to_s.empty? || article_id.to_s.empty?

      record = GoogleScholarCitationsStore.for_site(context.registers[:site]).lookup(scholar_id, article_id)
      return "" unless record

      if @field == "details"
        date = Time.iso8601(record["fetched_at"]).utc.strftime("%Y-%m-%d")
        return "#{record['count']} Google Scholar citations; last retrieved #{date} (UTC)"
      end

      "#{record['count'] / 10 * 10}+"
    end
  end
end

Liquid::Template.register_tag("google_scholar_citations", Jekyll::GoogleScholarCitationsTag)
Jekyll::Hooks.register :site, :after_reset do |site|
  site.instance_variable_set(:@google_scholar_citations, nil)
end
Jekyll::Hooks.register :site, :post_write do |site|
  site.instance_variable_get(:@google_scholar_citations)&.write
end
